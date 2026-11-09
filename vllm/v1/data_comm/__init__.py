"""host-host 数据传输库(vllm 仓):与设备解耦的通用传输实现。

现行实现为 UCX(host RoCE):ucx_channel.py(UHC1 协议族,带 tag 的
字节管道);ucx_link.py 承载 UCP 链路生命周期,ucx_ucp.py/ucx_pool.py/
sock.py 为共享底层,ucx_port.py 提供 embedded/daemon 双模式端口,
ucx_daemon.py/ucx_shm.py 为守护进程模式。设备拷贝(D2H/H2D)编排属
调用方——NPU 实现在 vllm_ascend(lwd_data_plane/host 包),GPU/CPU 边侧
按同接口在各自插件仓实现。

数据面关停注册表见 ``vllm/v1/lwd_comm``。
"""
