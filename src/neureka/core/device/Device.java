package neureka.core.device;

import java.util.HashMap;
import java.util.List;

import com.aparapi.device.OpenCLDevice;
import neureka.core.T;

/**
 *
 */
public class Device {
    /**
     * _tensorsMap:
     * Holds REGISTER _pointers f tensors stored on the _device.
     */
    private HashMap<T, Integer> _tensorsMap = new HashMap<T, Integer>();
    /**
     * REGISTER:
     * Maps REGISTER _pointers to _pointers WITHIN the compute _device.
     * Pointers within the _kernel change dynamically,
     * whereas the REGISTER entry will always represent a specific tensor from
     * the time of allocation to tensor deletion and de-allocation on the _device.
     */
    private int[][] _register = null;
    private OpenCLDevice _device;
    private TensorKernel _kernel;

    public Device(String name) {
        if (name == null) {
            _device = null;
            _kernel = null;
        } else {
            String[] parts = name.split(" ");
            String type = "gpu";
            if (parts != null && parts.length > 1) {
                type = parts[1];
                name = parts[0];
            }
            _register = new int[][]{new int[]{-1, -1, -1, -1, -1}};
            _device = OpenCLDevice.listDevices(null).get(0);
            List<OpenCLDevice> OpenCLDevices = OpenCLDevice.listDevices(null);
            for (OpenCLDevice found : OpenCLDevices) {
                System.out.println("\n---\n" + found.toString());
                System.out.println(found.getShortDescription() + "; ID: " + found.getDeviceId());
                if (
                        (found.getShortDescription() + found.toString()).toLowerCase().contains(name.toLowerCase())
                                && found.getType().toString().toLowerCase().contains(type)
                ) {
                    _device = found;
                }
            }
            if (!_device.getType().toString().toLowerCase().contains("cpu")) {
                _device.setSharedMemory(false);// GPU's (!cpu's) don't share host memory!
            }
            System.out.println("\nChosen _device:\n------------\n" + _device.toString() + "\n------------\n");
            System.out.println("\n_device _f_id:\n------------\n" + _device.getType().toString() + "\n------------\n");
            _kernel = new TensorKernel();
            System.out.println("Device f _kernel:\n------------");
            System.out.println(_kernel.getTargetDevice().toString());
        }
    }

    public void dispose() {
        _kernel.dispose();
    }

    public TensorKernel getKernel() {
        //System.out.println(_kernel.cleanUpArrays());
        return _kernel;
    }

    public boolean has(T tensor) {
        return _tensorsMap.containsKey(tensor);
    }

    /**
     * ======================================================================
     */
    public void get(T tensor) {
        if (_kernel != null) {
            _kernel.execute(
                    _device.createRange(
                            _kernel.executionSizeOf_fetchTsr(_register[0][_tensorsMap.get(tensor)], false)
                    )
            );
            T.factory.inject(_kernel.value(), false, tensor);
            if (tensor.rqsGradient()) {
                _kernel.execute(
                        _device.createRange(
                                _kernel.executionSizeOf_fetchTsr(_register[0][_tensorsMap.get(tensor)], true)
                        )
                );
                T.factory.inject(_kernel.value(), true, tensor);
            }
            rmv(tensor);
        }
    }

    public void rmv(T tensor) {
        if (_kernel != null) {
            if (_tensorsMap.containsKey(tensor)) {
                _kernel.freePtrOf(_tensorsMap.get(tensor), _register);
                _tensorsMap.remove(tensor);
                tensor.setIsOutsourced(false);
            }
        }
    }

    public Device add(T tensor) {
        tensor.setIsVirtual(false);
        if (!_tensorsMap.containsKey(tensor)) {
            _tensorsMap.put(tensor,
                    _kernel.allocPtrFor(tensor, _register)
            );
        }
        _kernel.execute(
                _device.createRange(
                        _kernel.executionSizeOf_storeTsr(_register[0][_tensorsMap.get(tensor)], tensor.value(), false)
                )
        );
        if (tensor.rqsGradient()) {
            double[] grd = (tensor.gradient() == null) ? new double[tensor.value().length] : tensor.gradient();
            _kernel.execute(
                    _device.createRange(
                            _kernel.executionSizeOf_storeTsr(
                                    _register[0][_tensorsMap.get(tensor)],
                                    grd, true
                            )
                    )
            );
        }
        tensor.add(this);
        tensor.setIsOutsourced(true);
        return this;
    }

    public double[] valueOf(T tensor, boolean grd) {
        _kernel.execute(
                _device.createRange(
                        _kernel.executionSizeOf_fetchTsr(_register[0][_tensorsMap.get(tensor)], grd)
                )
        );
        return _kernel.value();
    }

    public void calculate(T[] tsrs, int f_id, int d) {
        if (_kernel == null) {
            //What then?
        } else {
            int[] mode = new int[tsrs.length + 2];
            mode[0] = f_id;
            mode[mode.length - 1] = d;
            for (int mi = 0; mi < (tsrs.length); mi++) {
                mode[mi + 1] = (tsrs[mi] != null) ? _register[0][_tensorsMap.get(tsrs[mi])] : -1;
            }
            _kernel.execute(
                    _device.createRange(
                            _kernel.executionSizeOf_calc(mode)
                    )
            );
        }

    }

    public void calculate(T t, double value, int f_id) {
        if (_kernel == null) {
            //What then?
        } else {
            int[] mode = new int[2];
            mode[0] = f_id;
            mode[1] = _register[0][_tensorsMap.get(t)];
            _kernel.execute(
                    _device.createRange(
                            _kernel.executionSizeOf_calc(mode, value)
                    )
            );
        }
    }

    public void calculate_on_CPU(T drn, T t1, T t2, int f_id, int d) {
        System.out.println("TEST STARTS:");
        System.out.println(stringified(_kernel.values()));
        System.out.println(stringified(_kernel.pointers()));
        System.out.println(stringified(_kernel.shapes()));
        System.out.println(stringified(_kernel.translations()));
        int[] m;
        if (f_id < 7) {
            m = new int[]{f_id, _register[0][_tensorsMap.get(drn)], _register[0][_tensorsMap.get(t1)], (t2 != null) ? _register[0][_tensorsMap.get(t2)] : d};
        } else if (f_id < 12) {
            m = new int[]{f_id, _register[0][_tensorsMap.get(drn)], _register[0][_tensorsMap.get(t1)], d};
        } else {
            m = new int[]{f_id, _register[0][_tensorsMap.get(drn)], _register[0][_tensorsMap.get(t1)], _register[0][_tensorsMap.get(t2)], d};
        }
        int size = _kernel.executionSizeOf_calc(m);
        System.out.println("size: " + size);
        //_kernel._mde = m;
        for (int i = 0; i < size; i++) {
            _kernel.run(i, m);
            _kernel._idx = new int[_kernel._idx.length];
        }
        printDeviceContent(false);
    }

    public void printDeviceContent(boolean fetch) {
        if (fetch) {
            System.out.println(stringified(_kernel.values()));
            System.out.println(stringified(_kernel.pointers()));
            System.out.println(stringified(_kernel.shapes()));
            System.out.println(stringified(_kernel.translations()));
            System.out.println(stringified(_kernel.idx()));
        } else {
            System.out.println(stringified(_kernel._values));
            System.out.println(stringified(_kernel._pointers));
            System.out.println(stringified(_kernel._shapes));
            System.out.println(stringified(_kernel._translations));
            System.out.println(stringified(_kernel._idx));

        }

    }


    public String stringified(double[] a) {
        String result = "";
        for (double ai : a) {
            result += ai + ", ";
        }
        return result;
    }

    public String stringified(int[] a) {
        String result = "";
        for (int ai : a) {
            result += ai + ", ";
        }
        return result;
    }

}

