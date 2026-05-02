SHELL := cmd.exe
.SHELLFLAGS := /C

EXE ?= sycl_app.exe
RUNTIME_DIR ?= runtime
REAL_EXE ?= $(RUNTIME_DIR)/sycl_app_real.exe
MKLROOT ?= C:/Program Files (x86)/Intel/oneAPI/mkl/latest
SYCL_TARGETS ?= spir64_x86_64,nvptx64-nvidia-cuda

MAKEFLAGS += --no-print-directory --silent
ifeq ($(origin CXX), default)
CXX = clang++
endif
CXXDIR = $(shell powershell -NoProfile -Command "$$cmd = Get-Command '$(CXX)' -ErrorAction Stop | Select-Object -First 1 -ExpandProperty Source; [System.IO.Path]::GetDirectoryName($$cmd) + [System.IO.Path]::DirectorySeparatorChar" 2>NUL)
APP_CXXFLAGS = -O3 -fsycl -fsycl-targets=$(SYCL_TARGETS) -I"$(MKLROOT)/include"
APP_LDFLAGS = "$(RUNTIME_DIR)/mkl_rt.lib"

APP_SRC = $(filter-out launcher.cpp,$(wildcard *.cpp))
LAUNCHER_SRC = launcher.cpp

all: $(EXE) $(REAL_EXE)

$(RUNTIME_DIR):
	powershell -NoProfile -Command "if (!(Test-Path '$(RUNTIME_DIR)')) { New-Item -ItemType Directory -Path '$(RUNTIME_DIR)' | Out-Null }"

$(RUNTIME_DIR)/mkl_rt.lib: | $(RUNTIME_DIR)
	powershell -NoProfile -Command "Copy-Item -Force '$(MKLROOT)/lib/mkl_rt.lib' '$(RUNTIME_DIR)/'"

$(REAL_EXE): $(APP_SRC) | $(RUNTIME_DIR) $(RUNTIME_DIR)/mkl_rt.lib
	$(CXX) $(APP_CXXFLAGS) $(APP_SRC) -o $(REAL_EXE) $(APP_LDFLAGS)
	powershell -NoProfile -Command "Copy-Item -Force '$(MKLROOT)/bin/mkl_rt.2.dll','$(MKLROOT)/bin/mkl_core.2.dll','$(MKLROOT)/bin/mkl_def.2.dll','$(MKLROOT)/bin/mkl_avx2.2.dll','$(MKLROOT)/bin/mkl_avx512.2.dll','$(MKLROOT)/bin/mkl_mc3.2.dll','$(MKLROOT)/bin/mkl_intel_thread.2.dll','$(MKLROOT)/bin/mkl_sequential.2.dll','C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin/libiomp5md.dll','C:/Program Files (x86)/Intel/oneAPI/tbb/latest/bin/tbb12.dll' '$(RUNTIME_DIR)/'"
	powershell -NoProfile -Command "Copy-Item -Force '$(CXXDIR)sycl9.dll','$(CXXDIR)sycl-jit.dll','$(CXXDIR)ur_loader.dll','$(CXXDIR)ur_win_proxy_loader.dll','$(CXXDIR)ur_adapter_opencl.dll','$(CXXDIR)ur_adapter_cuda.dll','$(CXXDIR)ur_adapter_level_zero.dll','$(CXXDIR)ur_adapter_level_zero_v2.dll','$(CXXDIR)OpenCL.dll' '$(RUNTIME_DIR)/'"

$(EXE): $(LAUNCHER_SRC)
	$(CXX) -O2 $(LAUNCHER_SRC) -o $(EXE) -municode -lkernel32

run:
	.\$(EXE)

clean:
	powershell -NoProfile -Command "Remove-Item -Force '.\\$(EXE)','.\\sycl_app.lib','.\\sycl_app.exp' -ErrorAction SilentlyContinue; Remove-Item -Recurse -Force '.\\$(RUNTIME_DIR)','.\\sycl_app.exe.local' -ErrorAction SilentlyContinue; exit 0"
	del /F /Q *.obj 2>nul
	del /F /Q *.o 2>nul

# One-time helper to clean up DLL/lib leftovers from the old flat layout.
clean-flat:
	powershell -NoProfile -Command "Remove-Item -Force '.\\mkl_rt.2.dll','.\\mkl_core.2.dll','.\\mkl_def.2.dll','.\\mkl_avx2.2.dll','.\\mkl_avx512.2.dll','.\\mkl_mc3.2.dll','.\\mkl_intel_thread.2.dll','.\\mkl_sequential.2.dll','.\\libiomp5md.dll','.\\tbb12.dll','.\\sycl9.dll','.\\sycl-jit.dll','.\\ur_loader.dll','.\\ur_win_proxy_loader.dll','.\\ur_adapter_opencl.dll','.\\ur_adapter_cuda.dll','.\\ur_adapter_level_zero.dll','.\\ur_adapter_level_zero_v2.dll','.\\OpenCL.dll','.\\mkl_rt.lib' -ErrorAction SilentlyContinue; exit 0"

full:
	@$(MAKE) clean
	@$(MAKE) all
	@$(MAKE) run
