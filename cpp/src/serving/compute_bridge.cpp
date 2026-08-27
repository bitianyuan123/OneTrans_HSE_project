#include "serving/compute_bridge.h"

#include <limits.h>
#include <stdlib.h>
#include <unistd.h>

#include <chrono>
#include <stdexcept>

#if defined(ONETRANS_WITH_PYTHON)
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#endif

namespace onetrans {

#if defined(ONETRANS_WITH_PYTHON)

// --------------------------------------------------------------------------- //
// 初始化（start 线程内执行；成功后置 ready_）
// --------------------------------------------------------------------------- //
bool PythonComputeBridge::init_python(const std::string& module_dir,
                                      const std::string& weights_dir) {
    // Py_Initialize 线程安全且幂等；首次调用线程自此持有 GIL。初始化全程在
    // 本线程执行；首轮结束时 PyEval_SaveThread 释放 GIL 并注销线程状态——
    // 否则本（detached）线程退出后 GIL 仍被持有，run 线程的 Ensure 将永久
    // 阻塞。重试轮（GIL 已释放）走 Ensure/Release 配对。
    Py_Initialize();
    const bool first_round = !py_thread_saved_;
    PyGILState_STATE gil = PyGILState_STATE(0);
    if (!first_round) gil = PyGILState_Ensure();  // 重试轮：GIL 已释放，需显式取
    bool ok = false;
    do {
        // 归一化绝对路径（相对路径会破坏向上探测）
        char abs_buf[PATH_MAX];
        std::string module_abs =
            ::realpath(module_dir.c_str(), abs_buf) ? abs_buf : module_dir;
        // sys.path 注入：bridge_score.py 目录 + 向上探测仓库根（onetrans 包所在）
        std::string repo_root = module_abs;
        for (int i = 0; i < 6 && !repo_root.empty(); ++i) {
            if (::access((repo_root + "/onetrans/__init__.py").c_str(), F_OK) == 0) break;
            size_t slash = repo_root.find_last_of('/');
            if (slash == std::string::npos || slash == 0) {
                repo_root = slash == 0 ? "/" : "";
                if (::access((repo_root + "onetrans/__init__.py").c_str(), F_OK) != 0)
                    repo_root.clear();
                break;
            }
            repo_root = repo_root.substr(0, slash);
        }
        std::string add_path = "import sys; sys.path.insert(0, r'" + module_abs + "')";
        if (!repo_root.empty())
            add_path += "; sys.path.insert(0, r'" + repo_root + "')";
        if (PyRun_SimpleString(add_path.c_str()) != 0) {
            last_error_ = "sys.path 注入失败";
            break;
        }
        PyObject* mod = PyImport_ImportModule("bridge_score");
        if (!mod) {
            PyErr_Print();
            last_error_ = "import bridge_score 失败（检查 module_dir / onetrans 包）";
            break;
        }
        PyObject* init_fn = PyObject_GetAttrString(mod, "init");
        PyObject* score_fn = PyObject_GetAttrString(mod, "score_batch");
        PyObject* meta_fn = PyObject_GetAttrString(mod, "meta");
        if (!init_fn || !score_fn || !meta_fn) {
            last_error_ = "bridge_score 缺少 init/score_batch/meta";
            Py_XDECREF(init_fn);
            Py_XDECREF(score_fn);
            Py_XDECREF(meta_fn);
            Py_DECREF(mod);
            break;
        }
        // init(weights_dir)：import torch + 构建 OneTrans + 权重上 GPU（耗时，持 GIL）
        PyObject* args = Py_BuildValue("(s)", weights_dir.c_str());
        PyObject* meta = PyObject_CallObject(init_fn, args);
        Py_XDECREF(args);
        Py_DECREF(init_fn);
        if (!meta) {
            PyErr_Print();
            last_error_ = "bridge_score.init 失败（torch/权重加载）";
            Py_DECREF(score_fn);
            Py_DECREF(meta_fn);
            Py_DECREF(mod);
            break;
        }
        Py_DECREF(meta);
        Py_DECREF(mod);

        py_module_ = nullptr;  // mod 不长期持有（函数已 incref）
        py_score_fn_ = score_fn;
        py_meta_fn_ = meta_fn;
        ok = true;
    } while (false);
    if (first_round) {
        // 首轮：释放 GIL 并注销本线程的线程状态（此后本线程不再触碰 Python C API）
        PyEval_SaveThread();
        py_thread_saved_ = true;
    } else {
        PyGILState_Release(gil);
    }
    return ok;
}

bool PythonComputeBridge::start(const std::string& module_dir, const std::string& weights_dir,
                                int init_timeout_s) {
    if (ready_.load(std::memory_order_acquire)) return true;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!stopped_) return false;  // 已在启动中
        stopped_ = false;
    }
    // 初始化在独立线程执行（import torch 可达数十秒，不阻塞调用线程）
    std::thread init_thread([this, module_dir, weights_dir] {
        if (init_python(module_dir, weights_dir)) {
            ready_.store(true, std::memory_order_release);
        } else {
            std::lock_guard<std::mutex> lk(mu_);
            stopped_ = true;
        }
        cv_.notify_all();
    });
    init_thread.detach();

    // 等待初始化完成或超时（超时降级：ready_ 仍可能稍后置位，无害）
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::seconds(init_timeout_s > 0 ? init_timeout_s : 120);
    std::unique_lock<std::mutex> lk(mu_);
    while (!cv_.wait_until(lk, deadline,
                           [this] { return stopped_ || ready_.load(std::memory_order_acquire); })) {
        if (std::chrono::steady_clock::now() >= deadline) {
            last_error_ = "bridge 初始化超时（import torch / 权重上载）";
            return false;
        }
    }
    if (!ready_.load(std::memory_order_acquire)) return false;  // init 失败
    lk.unlock();

    // 桥工作线程（唯一持 GIL 的计算线程）
    thread_ = std::thread([this] { run(); });
    return true;
}

void PythonComputeBridge::stop() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (stopped_ && !thread_.joinable()) return;
        stopped_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    // 释放 Python 函数引用（桥线程已 join，GIL 安全）
#if defined(ONETRANS_WITH_PYTHON)
    if (py_score_fn_ || py_meta_fn_) {
        PyGILState_STATE gil = PyGILState_Ensure();
        Py_XDECREF(reinterpret_cast<PyObject*>(py_score_fn_));
        Py_XDECREF(reinterpret_cast<PyObject*>(py_meta_fn_));
        py_score_fn_ = nullptr;
        py_meta_fn_ = nullptr;
        PyGILState_Release(gil);
    }
#endif
}

bool PythonComputeBridge::submit(BridgeBatch batch) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (stopped_ || !ready_.load(std::memory_order_acquire)) return false;
        if (q_.size() >= cap_) return false;
        q_.push_back(std::move(batch));
    }
    cv_.notify_one();
    return true;
}

size_t PythonComputeBridge::queued() const {
    std::lock_guard<std::mutex> lk(mu_);
    return q_.size();
}

// --------------------------------------------------------------------------- //
// 桥线程主循环：C++ 队列 pop（不持 GIL）→ Ensure → 前向 → Release → 回调
// --------------------------------------------------------------------------- //
void PythonComputeBridge::run() {
    while (true) {
        BridgeBatch batch;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this] { return stopped_ || !q_.empty(); });
            if (q_.empty()) return;  // stop 且 drain 完
            batch = std::move(q_.front());
            q_.pop_front();
        }
        try {
            std::string logits = call_score(batch);
            if (batch.on_ok) batch.on_ok(std::move(logits));
        } catch (const std::exception& e) {
            if (batch.on_fail) batch.on_fail(e.what());
        } catch (...) {
            if (batch.on_fail) batch.on_fail("bridge: 未知异常");
        }
    }
}

std::string PythonComputeBridge::call_score(BridgeBatch& b) {
    PyGILState_STATE gil = PyGILState_Ensure();
    std::string out;
    do {
        PyObject* fn = reinterpret_cast<PyObject*>(py_score_fn_);
        if (!fn) throw std::runtime_error("bridge: score_batch 未初始化");

        // 参数：kv_blobs: list[bytes], row_map: list[int], ns_blob: bytes,
        //       n_rows: int, ns: int, d: int
        PyObject* kv_list = PyList_New(static_cast<Py_ssize_t>(b.kv_payloads.size()));
        for (size_t i = 0; i < b.kv_payloads.size(); ++i) {
            PyObject* bytes = PyBytes_FromStringAndSize(
                b.kv_payloads[i].data(), static_cast<Py_ssize_t>(b.kv_payloads[i].size()));
            PyList_SetItem(kv_list, static_cast<Py_ssize_t>(i), bytes);  // 窃取引用
        }
        PyObject* row_list = PyList_New(static_cast<Py_ssize_t>(b.row_kv_idx.size()));
        for (size_t i = 0; i < b.row_kv_idx.size(); ++i)
            PyList_SetItem(row_list, static_cast<Py_ssize_t>(i),
                           PyLong_FromLong(b.row_kv_idx[i]));

        PyObject* ns_bytes =
            PyBytes_FromStringAndSize(b.ns_blob.data(), static_cast<Py_ssize_t>(b.ns_blob.size()));
        PyObject* args = Py_BuildValue("(OOOiii)", kv_list, row_list, ns_bytes,
                                        static_cast<int>(b.n_rows), static_cast<int>(b.ns),
                                        static_cast<int>(b.d_model));
        Py_DECREF(kv_list);
        Py_DECREF(row_list);
        Py_DECREF(ns_bytes);

        PyObject* res = PyObject_CallObject(fn, args);
        Py_DECREF(args);
        if (!res) {
            PyErr_Print();
            throw std::runtime_error("bridge: score_batch 调用失败");
        }
        char* buf = nullptr;
        Py_ssize_t len = 0;
        if (PyBytes_AsStringAndSize(res, &buf, &len) != 0) {
            Py_DECREF(res);
            throw std::runtime_error("bridge: score_batch 返回类型非 bytes");
        }
        out.assign(buf, static_cast<size_t>(len));
        Py_DECREF(res);
    } while (false);
    PyGILState_Release(gil);
    return out;
}

#else  // !ONETRANS_WITH_PYTHON

// 无 Python 支持的编译降级：恒不可用，上层走 C++ CPU score_ns_batch

bool PythonComputeBridge::start(const std::string&, const std::string&, int) {
    last_error_ = "构建未启用 Python 支持（-DONETRANS_WITH_PYTHON=OFF）";
    return false;
}
void PythonComputeBridge::stop() {}
bool PythonComputeBridge::submit(BridgeBatch) { return false; }
size_t PythonComputeBridge::queued() const { return 0; }

#endif

}  // namespace onetrans
