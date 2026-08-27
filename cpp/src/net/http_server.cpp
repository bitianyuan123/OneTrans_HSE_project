#include "net/http_server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <stdexcept>

namespace onetrans {

namespace {

constexpr size_t kMaxHead = 64 * 1024;   // 请求头上限
constexpr size_t kMaxBody = 16 * 1024 * 1024;  // body 上限 16MB

const char* status_text(int code) {
    switch (code) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 413: return "Payload Too Large";
        case 500: return "Internal Server Error";
        default: return "Unknown";
    }
}

// 简单fd队列：accept 线程投递，worker 线程消费
class FdQueue {
public:
    void push(int fd) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            q_.push(fd);
        }
        cv_.notify_one();
    }

    int pop() {  // 阻塞；stop 后返回 -1
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return !q_.empty() || stopped_; });
        if (q_.empty()) return -1;
        int fd = q_.front();
        q_.pop();
        return fd;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stopped_ = true;
        }
        cv_.notify_all();
    }

private:
    std::mutex mu_;
    std::condition_variable cv_;
    std::queue<int> q_;
    bool stopped_ = false;
};

}  // namespace

// --------------------------------------------------------------------------- //
// HttpRequest
// --------------------------------------------------------------------------- //
std::map<std::string, std::string> HttpRequest::query_params() const {
    std::map<std::string, std::string> out;
    size_t pos = 0;
    while (pos < query.size()) {
        size_t amp = query.find('&', pos);
        if (amp == std::string::npos) amp = query.size();
        if (amp > pos) {
            std::string kv = query.substr(pos, amp - pos);
            size_t eq = kv.find('=');
            if (eq == std::string::npos)
                out[kv] = "";
            else
                out[kv.substr(0, eq)] = kv.substr(eq + 1);
        }
        pos = amp + 1;
    }
    return out;
}

HttpResponse HttpResponse::json(int status, std::string body) {
    HttpResponse r;
    r.status = status;
    r.content_type = "application/json";
    r.body = std::move(body);
    return r;
}

HttpResponse HttpResponse::text(int status, std::string body) {
    HttpResponse r;
    r.status = status;
    r.content_type = "text/plain; charset=utf-8";
    r.body = std::move(body);
    return r;
}

// --------------------------------------------------------------------------- //
// HttpServer
// --------------------------------------------------------------------------- //
HttpServer::HttpServer(const std::string& host, int port, int num_threads)
    : host_(host), port_(port), num_threads_(num_threads > 0 ? num_threads : 4) {}

HttpServer::~HttpServer() { stop(); }

void HttpServer::route(const std::string& method, const std::string& path, HttpHandler handler) {
    routes_[method + " " + path] = std::move(handler);
}

void HttpServer::route_async(const std::string& method, const std::string& path,
                             AsyncHandler handler) {
    async_routes_[method + " " + path] = std::move(handler);
}

void HttpServer::run() {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) throw std::runtime_error("socket() 失败");

    int opt = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port_));
    if (::inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) != 1)
        throw std::runtime_error("非法监听地址: " + host_);
    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0)
        throw std::runtime_error("bind 失败: " + host_ + ":" + std::to_string(port_));
    if (::listen(listen_fd_, 512) < 0) throw std::runtime_error("listen 失败");

    // 本进程单 server 实例：accept 线程 → worker 池（static 存储期，无需捕获）
    static FdQueue queue;
    running_ = true;
    for (int i = 0; i < num_threads_; ++i) {
        threads_.emplace_back([this] {
            while (running_) {
                int fd = queue.pop();
                if (fd < 0) break;  // stop
                handle_connection(fd);
            }
        });
    }

    // accept 循环（run 调用线程执行；返回即停止）
    while (running_) {
        sockaddr_in peer{};
        socklen_t plen = sizeof(peer);
        int fd = ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&peer), &plen);
        if (fd < 0) {
            if (errno == EINTR) continue;
            break;
        }
        int one = 1;
        ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        queue.push(fd);
    }
    queue.stop();
}

void HttpServer::stop() {
    if (!running_.exchange(false)) return;
    if (listen_fd_ >= 0) {
        ::shutdown(listen_fd_, SHUT_RDWR);
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
    for (auto& t : threads_) {
        if (t.joinable()) t.join();
    }
    threads_.clear();
    // drain 在途异步请求（完成线程会写回并关闭 fd；上限 10s 防御性退出）
    for (int i = 0; i < 10000 && pending_async_.load() > 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

// 读到 "\r\n\r\n" 为止（请求头），继续读 Content-Length 字节 body
bool HttpServer::read_request(int fd, HttpRequest& req, std::string& raw_head) {
    std::string buf;
    buf.reserve(2048);
    char chunk[4096];
    size_t head_end = std::string::npos;
    while (buf.size() < kMaxHead) {
        ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) return false;
        buf.append(chunk, static_cast<size_t>(n));
        head_end = buf.find("\r\n\r\n");
        if (head_end != std::string::npos) break;
    }
    if (head_end == std::string::npos) return false;
    raw_head = buf.substr(0, head_end);
    std::string rest = buf.substr(head_end + 4);

    // request line: METHOD /path?query HTTP/1.1
    size_t line_end = raw_head.find("\r\n");
    std::string reqline = raw_head.substr(0, line_end);
    size_t sp1 = reqline.find(' ');
    size_t sp2 = reqline.rfind(' ');
    if (sp1 == std::string::npos || sp2 == sp1) return false;
    req.method = reqline.substr(0, sp1);
    std::string target = reqline.substr(sp1 + 1, sp2 - sp1 - 1);
    size_t q = target.find('?');
    if (q == std::string::npos) {
        req.path = target;
    } else {
        req.path = target.substr(0, q);
        req.query = target.substr(q + 1);
    }

    // headers：Content-Length
    size_t content_len = 0;
    size_t pos = line_end + 2;
    while (pos < raw_head.size()) {
        size_t he = raw_head.find("\r\n", pos);
        if (he == std::string::npos) he = raw_head.size();
        std::string line = raw_head.substr(pos, he - pos);
        pos = he + 2;
        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = line.substr(0, colon);
        std::string val = line.substr(colon + 1);
        while (!val.empty() && (val.front() == ' ' || val.front() == '\t')) val.erase(0, 1);
        for (auto& c : key) c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));
        if (key == "content-length") content_len = static_cast<size_t>(::atoll(val.c_str()));
    }
    if (content_len > kMaxBody) return false;

    // body：rest 已读部分 + 剩余
    req.body.reserve(content_len);
    req.body = rest.substr(0, std::min(rest.size(), content_len));
    while (req.body.size() < content_len) {
        ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) return false;
        size_t take = std::min(static_cast<size_t>(n), content_len - req.body.size());
        req.body.append(chunk, take);
    }
    return true;
}

void HttpServer::write_response(int fd, const HttpResponse& resp) {
    std::string head = "HTTP/1.1 " + std::to_string(resp.status) + " " + status_text(resp.status) +
                       "\r\nContent-Type: " + resp.content_type +
                       "\r\nContent-Length: " + std::to_string(resp.body.size()) +
                       "\r\nConnection: close\r\n\r\n";
    ::send(fd, head.data(), head.size(), MSG_NOSIGNAL);
    if (!resp.body.empty()) ::send(fd, resp.body.data(), resp.body.size(), MSG_NOSIGNAL);
    ::shutdown(fd, SHUT_RDWR);
    ::close(fd);
}

void HttpServer::handle_connection(int fd) {
    HttpRequest req;
    std::string raw_head;
    if (!read_request(fd, req, raw_head)) {
        ::close(fd);
        return;
    }

    const std::string key = req.method + " " + req.path;

    // 异步路由：接入线程提交后立即返回（不占 worker 等待计算，§7.4.3 接入边界）；
    // 响应由流水线完成线程经 done 写回。done 单次生效（原子标记防重复写）。
    auto ait = async_routes_.find(key);
    if (ait != async_routes_.end()) {
        pending_async_.fetch_add(1);
        auto done = [this, fd, called = std::make_shared<std::atomic<bool>>(false)](
                        HttpResponse resp) {
            if (called->exchange(true)) return;  // 幂等：只写一次
            write_response(fd, resp);
            pending_async_.fetch_sub(1);
        };
        try {
            ait->second(req, done);
        } catch (const std::exception& e) {
            done(HttpResponse::json(500, std::string("{\"error\":\"") + e.what() + "\"}"));
        } catch (...) {
            done(HttpResponse::json(500, "{\"error\":\"unknown\"}"));
        }
        return;  // worker 立即处理下一连接
    }

    HttpResponse resp;
    auto it = routes_.find(key);
    if (it != routes_.end()) {
        try {
            resp = it->second(req);
        } catch (const std::exception& e) {
            resp = HttpResponse::json(500, std::string("{\"error\":\"") + e.what() + "\"}");
        } catch (...) {
            resp = HttpResponse::json(500, "{\"error\":\"unknown\"}");
        }
    } else {
        // 存在路径但方法不符 → 405（同步与异步路由都算路径存在）
        bool path_exists = false;
        for (const auto& [k, h] : routes_) {
            if (k.substr(k.find(' ') + 1) == req.path) path_exists = true;
        }
        for (const auto& [k, h] : async_routes_) {
            if (k.substr(k.find(' ') + 1) == req.path) path_exists = true;
        }
        resp = path_exists ? HttpResponse::json(405, "{\"error\":\"method not allowed\"}")
                           : HttpResponse::json(404, "{\"error\":\"not found\"}");
    }

    write_response(fd, resp);
}

}  // namespace onetrans
