package neureka.core.device;

import java.util.HashMap;
import java.util.List;

import com.aparapi.Range;
import com.aparapi.device.OpenCLDevice;
import neureka.core.Tsr;
import neureka.core.function.IFunction;

/**
 *
 */
public class Device {
    /**
     * _tensorsMap:
     * Holds REGISTER _pointers f tensors stored on the _device.
     */
    private HashMap<Tsr, Integer> _tensorsMap = new HashMap<Tsr, Integer>();
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
                System.out.println("\n---\n" + found.toString()+ "\n---\n");
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
            _kernel = new TensorKernel();
            System.out.println("Device of new kernel:\n------------");
            System.out.println(_kernel.getTargetDevice().toString());
        }
    }

    public void dispose() {
        _kernel.dispose();
    }

    public TensorKernel getKernel() {
        return _kernel;
    }

    public boolean has(Tsr tensor) {
        return _tensorsMap.containsKey(tensor);
    }

    /**
     * ======================================================================
     */
    public void get(Tsr tensor) {
        if (_kernel != null) {
            _kernel.execute(
                    _device.createRange(
                            _kernel.executionSizeOf_fetchTsr(_register[0][_tensorsMap.get(tensor)], false)
                    )
            );
            Tsr.factory.inject(_kernel.value(), false, tensor);
            if (tensor.rqsGradient()) {
                _kernel.execute(
                        Range.create(
                                _kernel.executionSizeOf_fetchTsr(_register[0][_tensorsMap.get(tensor)], true)
                        )
                );
                Tsr.factory.inject(_kernel.value(), true, tensor);
            }
            rmv(tensor);
        }
    }

    public void rmv(Tsr tensor) {
        if (_kernel != null) {
            if (_tensorsMap.containsKey(tensor)) {
                _kernel.freePtrOf(_tensorsMap.get(tensor), _register);
                _tensorsMap.remove(tensor);
                tensor.setIsOutsourced(false);
            }
        }
    }

    public Device overwrite(Tsr drain, Tsr source){
        this.calculate(new Tsr[]{drain, source}, IFunction.TYPES.LOOKUP.get("idy"), -1);
        return this;
    }

    public Device overwrite(Tsr tensor, double[] value){
        boolean targetGradient = tensor.gradientIsTargeted();
        if(tensor.rqsGradient()){
            if(_tensorsMap.containsKey(tensor)){
                _kernel.execute(
                        Range.create(
                                _kernel.executionSizeOf_storeTsr(
                                        _register[0][_tensorsMap.get(tensor)],
                                        value,
                                        targetGradient
                                )
                        )
                );
            }
        }
        return this;
    }

    public Device add(Tsr tensor) {
        tensor.setIsVirtual(false);
        if (!_tensorsMap.containsKey(tensor)) {
            _tensorsMap.put(tensor, _kernel.allocPtrFor(tensor, _register));
            _kernel.execute(
                    Range.create(
                            _kernel.executionSizeOf_storeTsr(_register[0][_tensorsMap.get(tensor)], tensor.value(), false)
                    )
            );
            if (tensor.rqsGradient()) {
                double[] grd = (tensor.gradient() == null) ? new double[tensor.value().length] : tensor.gradient();
                _kernel.execute(
                        Range.create(
                                _kernel.executionSizeOf_storeTsr(
                                        _register[0][_tensorsMap.get(tensor)],
                                        grd, true
                                )
                        )
                );
            }
            tensor.add(this);
            tensor.setIsOutsourced(true);
        }
        return this;
    }

    public void swap(Tsr former, Tsr replacement){
        int ptr = _tensorsMap.get(former);
        _tensorsMap.remove(former);
        _tensorsMap.put(replacement, ptr);
        replacement.add(this);
        replacement.setIsOutsourced(true);
    }

    public double[] valueOf(Tsr tensor, boolean grd) {
        _kernel.execute(
                Range.create(
                        _kernel.executionSizeOf_fetchTsr(
                                _register[0][
                                        _tensorsMap.get(
                                                tensor
                                        )]
                                , grd
                        )
                )
        );
        return _kernel.value();
    }

    public void calculate(Tsr[] tsrs, int f_id, int d) {
        try{
            //_validate(tsrs);
            if (_kernel == null) {
                //What then?
            } else {
                int[] mode = new int[tsrs.length + 2];
                mode[0] = f_id;
                mode[mode.length - 1] = d;
                for (int mi = 0; mi < (tsrs.length); mi++) {
                    mode[mi + 1] = (tsrs[mi] != null) ? _register[0][_tensorsMap.get(tsrs[mi])] : -1;
                }
                byte gradPtrMod = 0;
                for(int i=0; i<tsrs.length; i++){
                    boolean duplicate = false;
                    for(int ii=i-1; ii>=0; ii--){
                        duplicate = (i!=ii && tsrs[ii]==tsrs[i])?true:duplicate;
                    }
                    if(tsrs[i].gradientIsTargeted() && !duplicate){
                        gradPtrMod += (1<<i);
                    }
                }
                _kernel.execute(
                        Range.create(
                                _kernel.executionSizeOf_calc(mode, gradPtrMod)
                        )
                );
                _kernel.resetGradPtr(gradPtrMod);
            }
        } catch (Exception e){
            System.out.println(e);
        }
    }

    private void _validate(Tsr[] tsrs){
        if(tsrs.length==2){
            if(tsrs[0]==tsrs[1] && tsrs[0].gradientIsTargeted()||tsrs[1].gradientIsTargeted()){
                throw new IllegalArgumentException("[Error]:( '->'/'<-' operator must not be applied to the same tensor! )");
            }
        }
    }

    public void calculate(Tsr t, double value, int f_id, int d) {
        //int d = -1;
        if (_kernel == null) {
            //What then?
        } else {
            if(d<0){
                int[] mode = new int[2];
                mode[0] = f_id;
                mode[1] = _register[0][_tensorsMap.get(t)];
                _kernel.execute(
                        _device.createRange(
                                _kernel.executionSizeOf_calc(mode, value, (byte) ((t.gradientIsTargeted())?1:0))
                        )
                );
                _kernel.resetGradPtr((byte) ((t.gradientIsTargeted())?1:0));
            } else {//Derivative:
                if(IFunction.TYPES.REGISTER[f_id]=="^"){
                    if(d==0){
                        this.calculate(t, value-1, IFunction.TYPES.LOOKUP.get("^"), -1);
                        this.calculate(t, value, IFunction.TYPES.LOOKUP.get("*"), -1);
                    } else {

                    }
                }

            }

        }

    }

    public void calculate_on_CPU(Tsr drn, Tsr t1, Tsr t2, int f_id, int d) {
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
        byte gradPtrMod = 0;
        gradPtrMod += (drn.gradientIsTargeted())?1:0;
        gradPtrMod += (t1.gradientIsTargeted())?2:0;
        gradPtrMod += (t2.gradientIsTargeted())?4:0;
        int size = _kernel.executionSizeOf_calc(m, gradPtrMod);
        _kernel.resetGradPtr(gradPtrMod);
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

    @Override
    public String toString(){
        return "["+_device.getDeviceId()+"]:("+_device.getShortDescription()+")";
    }

    public String toString(String m){
        switch (m){
            case "": return toString();
            case "s": return _device.getShortDescription();
            case "id": return "["+_device.getDeviceId()+"]:("+_device.getShortDescription()+")";
        }
        return _device.toString();
    }

}

