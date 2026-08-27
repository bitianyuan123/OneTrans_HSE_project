#include "common/json.h"

#include <cmath>
#include <cstring>
#include <sstream>

namespace onetrans {

// --------------------------------------------------------------------------- //
// 序列化
// --------------------------------------------------------------------------- //

static void dump_string(const std::string& s, std::ostringstream& os) {
    os << '"';
    for (char c : s) {
        switch (c) {
            case '"': os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    os << buf;
                } else {
                    os << c;
                }
        }
    }
    os << '"';
}

static void dump_number(double v, std::ostringstream& os) {
    if (std::isfinite(v) && v == std::floor(v) && std::fabs(v) < 1e15) {
        os << static_cast<long long>(v);
    } else {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.17g", v);
        os << buf;
    }
}

std::string JsonValue::dump() const {
    std::ostringstream os;
    switch (type_) {
        case Type::Null: os << "null"; break;
        case Type::Bool: os << (bool_ ? "true" : "false"); break;
        case Type::Number: dump_number(num_, os); break;
        case Type::String: dump_string(str_, os); break;
        case Type::Array: {
            os << '[';
            bool first = true;
            for (const auto& v : *arr_) {
                if (!first) os << ',';
                first = false;
                os << v.dump();
            }
            os << ']';
            break;
        }
        case Type::Object: {
            os << '{';
            bool first = true;
            for (const auto& [k, v] : *obj_) {
                if (!first) os << ',';
                first = false;
                dump_string(k, os);
                os << ':';
                os << v.dump();
            }
            os << '}';
            break;
        }
    }
    return os.str();
}

// --------------------------------------------------------------------------- //
// 解析（递归下降）
// --------------------------------------------------------------------------- //

namespace {

struct Parser {
    const char* p;
    const char* end;

    explicit Parser(const std::string& s) : p(s.data()), end(s.data() + s.size()) {}

    [[noreturn]] void fail(const std::string& msg) {
        throw std::runtime_error("json parse: " + msg);
    }

    void skip_ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
    }

    JsonValue parse() {
        skip_ws();
        if (p >= end) fail("empty");
        char c = *p;
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return JsonValue::str(parse_string());
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        return parse_number();
    }

    JsonValue parse_object() {
        JsonValue obj = JsonValue::object();
        ++p;  // {
        skip_ws();
        if (p < end && *p == '}') {
            ++p;
            return obj;
        }
        while (true) {
            skip_ws();
            if (p >= end || *p != '"') fail("expect key");
            std::string key = parse_string();
            skip_ws();
            if (p >= end || *p != ':') fail("expect ':'");
            ++p;
            obj.set(key, parse());
            skip_ws();
            if (p < end && *p == ',') {
                ++p;
                continue;
            }
            if (p < end && *p == '}') {
                ++p;
                return obj;
            }
            fail("expect ',' or '}'");
        }
    }

    JsonValue parse_array() {
        std::vector<JsonValue> items;
        ++p;  // [
        skip_ws();
        if (p < end && *p == ']') {
            ++p;
            return JsonValue::array(std::move(items));
        }
        while (true) {
            items.push_back(parse());
            skip_ws();
            if (p < end && *p == ',') {
                ++p;
                continue;
            }
            if (p < end && *p == ']') {
                ++p;
                return JsonValue::array(std::move(items));
            }
            fail("expect ',' or ']'");
        }
    }

    std::string parse_string() {
        ++p;  // "
        std::string out;
        while (p < end && *p != '"') {
            if (*p == '\\') {
                ++p;
                if (p >= end) fail("bad escape");
                switch (*p) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'u': {
                        if (end - p < 5) fail("bad \\u");
                        unsigned code = 0;
                        for (int i = 1; i <= 4; ++i) {
                            char h = p[i];
                            code <<= 4;
                            if (h >= '0' && h <= '9') code |= static_cast<unsigned>(h - '0');
                            else if (h >= 'a' && h <= 'f') code |= static_cast<unsigned>(h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') code |= static_cast<unsigned>(h - 'A' + 10);
                            else fail("bad hex");
                        }
                        p += 4;
                        // UTF-8 编码（BMP 内足够）
                        if (code < 0x80) {
                            out += static_cast<char>(code);
                        } else if (code < 0x800) {
                            out += static_cast<char>(0xC0 | (code >> 6));
                            out += static_cast<char>(0x80 | (code & 0x3F));
                        } else {
                            out += static_cast<char>(0xE0 | (code >> 12));
                            out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                            out += static_cast<char>(0x80 | (code & 0x3F));
                        }
                        break;
                    }
                    default: fail("bad escape char");
                }
                ++p;
            } else {
                out += *p++;
            }
        }
        if (p >= end || *p != '"') fail("unterminated string");
        ++p;
        return out;
    }

    JsonValue parse_bool() {
        if (end - p >= 4 && std::memcmp(p, "true", 4) == 0) {
            p += 4;
            return JsonValue::boolean(true);
        }
        if (end - p >= 5 && std::memcmp(p, "false", 5) == 0) {
            p += 5;
            return JsonValue::boolean(false);
        }
        fail("bad bool");
    }

    JsonValue parse_null() {
        if (end - p >= 4 && std::memcmp(p, "null", 4) == 0) {
            p += 4;
            return JsonValue::null();
        }
        fail("bad null");
    }

    JsonValue parse_number() {
        char* endptr = nullptr;
        double v = std::strtod(p, &endptr);
        if (endptr == p) fail("bad number");
        p = endptr;
        return JsonValue::number(v);
    }
};

}  // namespace

JsonValue json_parse(const std::string& text) {
    Parser parser(text);
    JsonValue v = parser.parse();
    return v;
}

}  // namespace onetrans
