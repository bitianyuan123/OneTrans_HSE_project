# 独立稀疏参数服务器（PS）构建文件（bazel）。
# 依赖 brpc / protobuf / glog / gflags（WORKSPACE 中以 git 拉取，标签见下）。
# 构建：bazel build //deploy/ps:embedding_server

load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_proto_library")

proto_library(
    name = "embedding_service_proto",
    srcs = ["embedding_service.proto"],
)

cc_proto_library(
    name = "embedding_service_cc_proto",
    deps = [":embedding_service_proto"],
)

cc_binary(
    name = "embedding_server",
    srcs = ["embedding_server.cc"],
    deps = [
        ":embedding_service_cc_proto",
        "@com_github_brpc_brpc//:brpc",
        "@com_github_google_glog//:glog",
        "@com_github_gflags_gflags//:gflags",
    ],
)