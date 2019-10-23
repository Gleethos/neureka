package neureka.core.device.aparapi;

import java.util.HashMap;
import java.util.List;

import com.aparapi.Range;
import com.aparapi.device.OpenCLDevice;
import neureka.core.Tsr;
import neureka.core.device.IDevice;
import neureka.core.function.IFunction;
import neureka.core.utility.DataHelper;

/**
 *
 */
public class AparapiDevice implements IDevice
{
    /**
     * _tensors_map:
     * Holds REGISTER _pointers f tensors stored on the _device.
     */
    private HashMap<Tsr, Integer> _tensors_map = new HashMap<Tsr, Integer>();
    /**
     * REGISTER:
     * Maps REGISTER _pointers to _pointers WITHIN the compute _device.
     * Pointers within the _kernel change dynamically,
     * whereas the REGISTER entry will always represent a specific tensor from
     * the time of allocation to tensor deletion and de-allocation on the _device.
     */
    private int[][] _register = null;
    private OpenCLDevice _device;
    private AbstractKernel _kernel;

    public AparapiDevice(String name)
    {
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
            name = name.toUpperCase();
            boolean useFP64 = false;
            useFP64 = useFP64 || name.contains("FP64") || name.contains("DOUBLE");
            _kernel = (useFP64)?new KernelFP64():new KernelFP32();
            System.out.println("AparapiDevice of new kernel:\n------------");
            System.out.println(_kernel.getTargetDevice().toString());
        }

    }

    public void dispose() {
        _kernel.dispose();
    }

    public AbstractKernel getKernel() {
        return _kernel;
    }

    public boolean has(Tsr tensor) {
        return _tensors_map.containsKey(tensor);
    }

    /**
     * ======================================================================
     */
    public IDevice get(Tsr tensor)
    {
        if (_kernel != null) {
            Tsr.fcn.inject(valueOf(tensor, false), false, tensor);//_kernel.value64(), false, tensor
            if (tensor.rqsGradient()) {
                Tsr.fcn.inject(valueOf(tensor, true), true, tensor);//_kernel.value64(), true, tensor
            }
            rmv(tensor);
        }
        return this;
    }

    public IDevice rmv(Tsr tensor)
    {
        if (_kernel != null) {
            if (_tensors_map.containsKey(tensor)) {
                _kernel.freePtrOf(_tensors_map.get(tensor), _register);
                _tensors_map.remove(tensor);
                tensor.setIsOutsourced(false);
            }
        }
        return this;
    }

    public IDevice overwrite(Tsr drain, Tsr source)
    {
        _execute(new Tsr[]{drain, source}, IFunction.TYPES.LOOKUP.get("idy"), -1);
        return this;
    }

    public IDevice overwrite(Tsr tensor, double[] value)
    {
        boolean targetGradient = tensor.gradientIsTargeted();
        if(tensor.rqsGradient()){
            if(_tensors_map.containsKey(tensor)){
                _kernel.execute(
                        Range.create(_device,
                                _kernel.executionSizeOf_storeTsr(
                                        _register[0][_tensors_map.get(tensor)],
                                        value,
                                        targetGradient
                                )
                        )
                );
            }
        }
        return this;
    }

    public IDevice add(Tsr tensor) {
        tensor.setIsVirtual(false);
        if (!_tensors_map.containsKey(tensor)) {
            _tensors_map.put(tensor, _kernel.allocPtrFor(tensor, _register));
            _kernel.execute(
                    Range.create(//_device, //-> Causes Error! Why?
                            _kernel.executionSizeOf_storeTsr(_register[0][_tensors_map.get(tensor)], tensor.value64(), false)
                    )
            );
            if (tensor.rqsGradient()) {
                double[] grd = (tensor.gradient64() == null) ? new double[tensor.value64().length] : tensor.gradient64();
                _kernel.execute(
                        Range.create(//_device, //-> Causes Error! Why?
                                _kernel.executionSizeOf_storeTsr(
                                        _register[0][_tensors_map.get(tensor)],
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

    public IDevice swap(Tsr former, Tsr replacement){
        int ptr = _tensors_map.get(former);
        _tensors_map.remove(former);
        _tensors_map.put(replacement, ptr);
        replacement.add(this);
        replacement.setIsOutsourced(true);
        return this;
    }

    public double[] valueOf(Tsr tensor, boolean grd) {
        _kernel.execute(
                Range.create(//_device, //-> Causes Error! Why?
                        _kernel.executionSizeOf_fetchTsr(
                                _register[0][
                                        _tensors_map.get(
                                                tensor
                                        )]
                                , grd
                        )
                )
        );
        return _kernel.value();
    }

    public float[] floatValueOf(Tsr tensor, boolean grd){
        return DataHelper.doubleToFloat(valueOf(tensor, grd));
    }

    public IDevice execute(Tsr[] tsrs, int f_id, int d)
    {
        if(IFunction.TYPES.REGISTER[f_id]=="<") {
            int offset = (tsrs[0]==null)?1:0;
            this._execute(new Tsr[]{tsrs[0+offset], tsrs[1+offset]}, IFunction.TYPES.LOOKUP.get("idy"), -1);
        } else if(IFunction.TYPES.REGISTER[f_id]==">") {
            int offset = (tsrs[0]==null)?1:0;
            _execute(new Tsr[]{tsrs[1+offset], tsrs[0+offset]}, IFunction.TYPES.LOOKUP.get("idy"), -1);
        } else {
            if(tsrs[0]==null){
                int[] shp =
                        (IFunction.TYPES.REGISTER[f_id] == "x")
                                ? Tsr.fcn.indexing.shpOfCon(tsrs[1].shape(), tsrs[2].shape()) : tsrs[1].shape();
                Tsr output = new Tsr(shp, 0.0);
                this.add(output);
                tsrs[0] = output;
            }
            if (
                    tsrs.length == 3
                            &&
                            (
                                    (tsrs[1].isVirtual() || tsrs[2].isVirtual())
                                            ||
                                    (!tsrs[1].isOutsourced() && tsrs[1].size() == 1 || !tsrs[2].isOutsourced() && tsrs[2].size() == 1)
                            )
            ) {
                if (tsrs[2].isVirtual() || tsrs[2].size() == 1) {
                    _execute(new Tsr[]{tsrs[0], tsrs[1]}, IFunction.TYPES.LOOKUP.get("idy"), -1);
                    _execute(tsrs[0], tsrs[2].value64()[0], f_id, d);
                } else {
                    _execute(new Tsr[]{tsrs[0], tsrs[2]}, IFunction.TYPES.LOOKUP.get("idy"), -1);
                    _execute(tsrs[0], tsrs[1].value64()[0], f_id, d);
                }
            } else {
                for (Tsr t : tsrs) {
                    if (!t.isOutsourced()) {
                        this.add(t);
                    }
                }
                _execute(tsrs, f_id, d);
            }
        }
        return this;
    }

    public void _execute(Tsr[] tsrs, int f_id, int d) {
        try{
            //_validate(tsrs);
            if (_kernel == null) {
                //What then?
            } else {
                int[] mode = new int[tsrs.length + 2];
                mode[0] = f_id;
                mode[mode.length - 1] = d;
                for (int mi = 0; mi < (tsrs.length); mi++) {
                    mode[mi + 1] = (tsrs[mi] != null) ? _register[0][_tensors_map.get(tsrs[mi])] : -1;
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
                        Range.create(//_device, //-> Causes Error! Why?
                                _kernel.executionSizeOf_calc(mode, gradPtrMod)
                        )
                );
                _kernel.closeExecution(gradPtrMod);
            }
        } catch (Exception e){
            System.out.println(e);
        }
    }

    //private void _validate(Tsr[] tsrs){
    //    if(tsrs.length==2){
    //        if(tsrs[0]==tsrs[1] && tsrs[0].gradientIsTargeted()||tsrs[1].gradientIsTargeted()){
    //            throw new IllegalArgumentException("[Error]:( '->'/'<-' operator must not be applied to the same tensor! )");
    //        }
    //    }
    //}

    public void _execute(Tsr t, double value, int f_id, int d)
    {
        if (_kernel == null) {
            //What then?
        } else {
            if(d<0){
                int[] mode = new int[2];
                mode[0] = f_id;
                mode[1] = _register[0][_tensors_map.get(t)];
                _kernel.execute(
                        Range.create(_device,
                                _kernel.executionSizeOf_calc(mode, value, (byte) ((t.gradientIsTargeted())?1:0))
                        )
                );
                _kernel.closeExecution((byte) ((t.gradientIsTargeted())?1:0));
            } else {
                /**   Derivatives implementation: (values cannot be derived)    **/
                if(
                    IFunction.TYPES.REGISTER[f_id]=="+"||
                    IFunction.TYPES.REGISTER[f_id]=="-"||
                    IFunction.TYPES.REGISTER[f_id]=="%"
                ){
                    _execute(t, 0, IFunction.TYPES.LOOKUP.get("*"), -1);
                    _execute(t, 1, IFunction.TYPES.LOOKUP.get("+"), -1);
                } else if(IFunction.TYPES.REGISTER[f_id]=="^"){
                    _execute(t, value-1, IFunction.TYPES.LOOKUP.get("^"), -1);
                    _execute(t, value, IFunction.TYPES.LOOKUP.get("*"), -1);
                } else if(IFunction.TYPES.REGISTER[f_id]=="*"){
                     _execute(t, 0, IFunction.TYPES.LOOKUP.get("*"), -1);
                     _execute(t, value, IFunction.TYPES.LOOKUP.get("+"), -1);
                } else if(IFunction.TYPES.REGISTER[f_id]=="/"){
                    _execute(t, 0, IFunction.TYPES.LOOKUP.get("*"), -1);
                    _execute(t, 1/value, IFunction.TYPES.LOOKUP.get("+"), -1);
                }

            }

        }

    }

    public void calculate_on_CPU(Tsr[] tsrs, int f_id, int d)
    {
        System.out.println("TEST STARTS:");
        System.out.println(stringified(_kernel.values()));
        System.out.println(stringified(_kernel.pointers()));
        System.out.println(stringified(_kernel.shapes()));
        System.out.println(stringified(_kernel.translations()));
        int[] m;
        byte gradPtrMod = 0;
        if(tsrs.length==3){
            Tsr drn = tsrs[0];
            Tsr t1 = tsrs[1];
            Tsr t2 = tsrs[2];

            if (f_id < 7) {
                m = new int[]{f_id, _register[0][_tensors_map.get(drn)], _register[0][_tensors_map.get(t1)], (t2 != null) ? _register[0][_tensors_map.get(t2)] : d};
            } else if (f_id < 12) {
                m = new int[]{f_id, _register[0][_tensors_map.get(drn)], _register[0][_tensors_map.get(t1)], d};
            } else {
                m = new int[]{f_id, _register[0][_tensors_map.get(drn)], _register[0][_tensors_map.get(t1)], _register[0][_tensors_map.get(t2)], d};
            }
            gradPtrMod += (drn.gradientIsTargeted())?1:0;
            gradPtrMod += (t1.gradientIsTargeted())?2:0;
            gradPtrMod += (t2.gradientIsTargeted())?4:0;
        } else {
            m = new int[tsrs.length+2];
            m[0] = f_id;
            m[m.length-1] = d;
            for(int i=1; i<m.length-1; i++){
                m[i] = _register[0][_tensors_map.get(tsrs[i-1])];
            }
        }
        int size = _kernel.executionSizeOf_calc(m, gradPtrMod);
        _kernel.closeExecution(gradPtrMod);
        System.out.println("size: " + size);
        //_kernel._mde = m;
        for (int i = 0; i < size; i++) {
            _kernel.run(i, m);
            ((KernelFP64)_kernel)._idx = new int[((KernelFP64)_kernel)._idx.length];
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
            //System.out.println(stringified(_kernel._values));
            //System.out.println(stringified(_kernel._pointers));
            //System.out.println(stringified(_kernel._shapes));
            //System.out.println(stringified(_kernel._translations));
            //System.out.println(stringified(_kernel._idx));

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
        return "["+_device.getDeviceId()+"]:("+_device.getShortDescription()+"-FP"+((_kernel instanceof  KernelFP64)?"64":32)+")";
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

