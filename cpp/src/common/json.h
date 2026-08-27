// 极简 JSON（object/array/string/number/bool/null），服务于：
// 权重 manifest 解析、HTTP 请求/响应体、KV payload header。
// 数字统一 double（id/offset 均 < 2^53，精确无损）。
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace onetrans {

class JsonValue {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    JsonValue() : type_(Type::Null) {}
    static JsonValue null() { return JsonValue(); }
    static JsonValue boolean(bool v) {
        JsonValue j;
        j.type_ = Type::Bool;
        j.bool_ = v;
        return j;
    }
    static JsonValue number(double v) {
        JsonValue j;
        j.type_ = Type::Number;
        j.num_ = v;
        return j;
    }
    static JsonValue str(std::string v) {
        JsonValue j;
        j.type_ = Type::String;
        j.str_ = std::move(v);
        return j;
    }
    static JsonValue array(std::vector<JsonValue> v) {
        JsonValue j;
        j.type_ = Type::Array;
        j.arr_ = std::make_shared<std::vector<JsonValue>>(std::move(v));
        return j;
    }
    static JsonValue object() {
        JsonValue j;
        j.type_ = Type::Object;
        j.obj_ = std::make_shared<std::map<std::string, JsonValue>>();
        return j;
    }

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::Null; }
    bool is_number() const { return type_ == Type::Number; }

    bool as_bool() const { return type_ == Type::Bool ? bool_ : false; }
    double as_number() const { return type_ == Type::Number ? num_ : 0.0; }
    int64_t as_int() const { return static_cast<int64_t>(num_); }
    const std::string& as_string() const { return str_; }
    const std::vector<JsonValue>& as_array() const {
        static const std::vector<JsonValue> kEmpty;
        return type_ == Type::Array ? *arr_ : kEmpty;
    }
    const std::map<std::string, JsonValue>& as_object() const {
        static const std::map<std::string, JsonValue> kEmpty;
        return type_ == Type::Object ? *obj_ : kEmpty;
    }

    // object 访问（缺失抛异常）
    const JsonValue& at(const std::string& key) const {
        if (type_ != Type::Object) throw std::runtime_error("json: at() on non-object");
        auto it = obj_->find(key);
        if (it == obj_->end()) throw std::runtime_error("json: 缺少键 " + key);
        return it->second;
    }
    void set(const std::string& key, JsonValue v) {
        if (type_ != Type::Object) throw std::runtime_error("json: set() on non-object");
        (*obj_)[key] = std::move(v);
    }
    void push_back(JsonValue v) {
        if (type_ != Type::Array) throw std::runtime_error("json: push_back() on non-array");
        arr_->push_back(std::move(v));
    }

    std::string dump() const;

private:
    Type type_;
    bool bool_ = false;
    double num_ = 0.0;
    std::string str_;
    std::shared_ptr<std::vector<JsonValue>> arr_;
    std::shared_ptr<std::map<std::string, JsonValue>> obj_;
};

JsonValue json_parse(const std::string& text);

}  // namespace onetrans
