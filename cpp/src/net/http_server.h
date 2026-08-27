// 极简 HTTP/1.1 接入层（Linux epoll-free 线程池模型）。
//
// 职责边界（对应设计文档 §4 接入层）：
// - TCP accept 循环 + 固定 worker 线程池处理连接；
// - 请求解析：request line + headers（Content-Length）+ body；
// - 路由注册：(method, path) → handler；未匹配返回 404；
// - 响应：Content-Length 显式声明，Connection: close（无 keep-alive，语义简单可靠）。
//
// 不做 TLS/压缩/分块编码——生产部署前置 LB（nginx/envoy）承担。
#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace onetrans {

struct HttpRequest {
    std::string method;  // GET / POST
    std::string path;     // 不含 query string
    std::string query;    // 原始 query（可空）
    std::string body;     // POST body

    // query 参数解析（?a=b&c=d → {"a":"b","c":"d"}）
    std::map<std::string, std::string> query_params() const;
};

struct HttpResponse {
    int status = 200;
    std::string content_type = "application/json";
    std::string body;

    static HttpResponse json(int status, std::string body);
    static HttpResponse text(int status, std::string body);
};

using HttpHandler = std::function<HttpResponse(const HttpRequest&)>;

class HttpServer {
public:
    HttpServer(const std::string& host, int port, int num_threads);
    ~HttpServer();

    // method: "GET" / "POST"（大小写敏感）
    void route(const std::string& method, const std::string& path, HttpHandler handler);

    // 阻塞运行（内部启动线程池 + accept 循环）；bind 失败抛异常
    void run();

    void stop();

private:
    void accept_loop();
    void handle_connection(int fd);
    bool read_request(int fd, HttpRequest& req, std::string& raw_head);

    std::string host_;
    int port_;
    int num_threads_;
    int listen_fd_ = -1;
    std::atomic<bool> running_{false};
    std::vector<std::thread> threads_;
    std::map<std::string, HttpHandler> routes_;  // key: "METHOD /path"
};

}  // namespace onetrans
