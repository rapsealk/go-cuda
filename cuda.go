package cuda

import (
	"encoding/hex"
	"errors"
)

// #cgo CFLAGS: -I/usr/local/cuda/include
// #cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcudart
// #include <cuda_runtime_api.h>
import "C"

type DeviceProp struct {
	Name           string
	UUID           string
	TotalGlobalMem uint32
}

var ErrCUDAInvalidValue = errors.New("This indicates that one or more..")

func GetDeviceCount() (int32, error) {
	var deviceCount C.int
	if C.cudaGetDeviceCount(&deviceCount) != 0 {
		return 0, ErrCUDAInvalidValue
	}
	return int32(deviceCount), nil
}

func GetDeviceProperties(device int32) (DeviceProp, error) {
	var prop C.struct_cudaDeviceProp
	if C.cudaGetDeviceProperties(&prop, C.int(device)) != 0 {
		return DeviceProp{}, ErrCUDAInvalidValue
	}
	c_uuid := []byte(C.GoString(&prop.uuid.bytes[0]))
	uuid := make([]byte, hex.EncodedLen(len(c_uuid)))
	hex.Encode(uuid, c_uuid)
	return DeviceProp{
		Name:           C.GoString(&prop.name[0]),
		UUID:           string(uuid),
		TotalGlobalMem: uint32(prop.totalGlobalMem),
	}, nil
}
