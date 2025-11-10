# py-videocore7

A Python library for GPGPU programming on Raspberry Pi 5, which realizes
assembling and running QPU programs.

For Raspberry Pi Zero/1/2/3, use
[Idein/py-videocore](https://github.com/Idein/py-videocore) instead.

For Raspberry Pi 4, use
[Idein/py-videocore6](https://github.com/Idein/py-videocore6) instead.

## About VideoCore VII QPU

Raspberry Pi 5 (BCM2712) has a GPU named VideoCore VII QPU in its SoC.
The basic instruction set (add/mul ALU dual issue, three delay slots et al.)
remains the same as VideoCore VI QPU of Raspberry Pi 4, and some units
now perform differently.

- VideoCore IV QPU @ 250MHz: 250 [MHz] x 3 [slice] x 4 [qpu/slice] x 4 [physical core/qpu] x 2 [op/cycle] = 24 [Gflop/s]
- VideoCore IV QPU @ 300MHz: 300 [MHz] x 3 [slice] x 4 [qpu/slice] x 4 [physical core/qpu] x 2 [op/cycle] = 28.8 [Gflop/s]
- VideoCore VI QPU @ 500MHz: 500 [MHz] x 2 [slice] x 4 [qpu/slice] x 4 [physical core/qpu] x 2 [op/cycle] = 32 [Gflop/s]
- VideoCore VII QPU @ 800MHz: 800 [MHz] x 3 [slice] x 4 [qpu/slice] x 4 [physical core/qpu] x 2 [op/cycle] = 76.8 [Gflop/s]


## Requirements

`py-videocore7` communicates with the V3D hardware through `/dev/dri/card0`,
which is exposed by the DRM V3D driver.
To access the device, you need to belong to `video` group or be `root` user.
If you choose the former, run `sudo usermod --append --groups video $USER`
(re-login to take effect).


## Installation

[Install `uv`](https://docs.astral.sh/uv/getting-started/installation/) and
clone `py-videocore7` and run with `uv`:

```console
$ sudo apt update
$ sudo apt upgrade
$ sudo apt install git
$ git clone https://github.com/Idein/py-videocore7.git
$ cd py-videocore7/
$ uv run examples/sgemm.py
```


## Running tests and examples

In the `py-videocore7` directory cloned above:

```console
$ uv run pytest -vs tests
```

```console
$ uv run examples/sgemm.py
==== sgemm example (1024x1024 times 1024x1024) ====
numpy: 0.0390 sec, 55.1937 Gflop/s
QPU:   0.1006 sec, 21.3827 Gflop/s
Minimum absolute error: 0.0
Maximum absolute error: 0.0003814697265625
Minimum relative error: 0.0
Maximum relative error: 0.13134673237800598
```

```console
$ uv run examples/scopy.py
==== CPU scopy example (24.0 Mi elements) ====
0.06235705600010988 sec, 1614.3048190059296 MB/s
==== QPU 1 thread scopy example (24.0 Mi elements) ====
Preparing for buffers...
Executing on QPU...
0.05958151100003306 sec, 1689.5055917588957 MB/s
==== QPU 12 threads scopy example (24.0 Mi elements) ====
Preparing for buffers...
Executing on QPU...
0.02430019499934133 sec, 4142.489227050586 MB/s
```

## References

- DRM V3D driver which controls QPU via hardware V3D registers: [linux/drivers/gpu/drm/v3d](https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git/tree/drivers/gpu/drm/v3d)
- Mesa library which partially includes the QPU instruction set: [mesa/src/broadcom/qpu](https://gitlab.freedesktop.org/mesa/mesa/-/tree/main/src/broadcom/qpu)
- py-videocore: [Idein/py-videocore](https://github.com/Idein/py-videocore)
- py-videocore6: [Idein/py-videocore6](https://github.com/Idein/py-videocore6)
