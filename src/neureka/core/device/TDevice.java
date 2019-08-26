package neureka.core.device;

import java.util.HashMap;
import java.util.List;

import com.aparapi.device.OpenCLDevice;
import neureka.core.T;

public class TDevice
{
    /**
     *    map:
     *    Holds REGISTER _pointers f tensors stored on the device.
    * */
    private HashMap<T, Integer> map = new HashMap<T, Integer>();
    /**
     *    REGISTER:
     *    Maps REGISTER _pointers to _pointers WITHIN the compute device.
     *    Pointers within the kernel change dynamically,
     *    whereas the REGISTER entry will always represent a specific tensor from
     *    the time of allocation to tensor deletion and de-allocation on the device.
     *
     * */
    private int[][] register = null;
    private OpenCLDevice device = null;
    //private Device device = null;

    private TKernel kernel = null;

    public TDevice(String name){
        if(name==null){
            device = null;
            kernel = null;
        }else{
            String[] parts = name.split(" ");
            String type = "gpu";
            if(parts!=null&&parts.length>1){
                type = parts[1];
                name = parts[0];
            }
            register = new int[][]{new int[]{-1,-1,-1,-1,-1}};
            device = OpenCLDevice.listDevices(null).get(0);
            List<OpenCLDevice> OpenCLDevices = OpenCLDevice.listDevices(null);
            for (OpenCLDevice found: OpenCLDevices){
                System.out.println("\n---\n"+found.toString());
                System.out.println(found.getShortDescription()+"; ID: "+found.getDeviceId());
                if(
                        (found.getShortDescription()+found.toString()).toLowerCase().contains(name.toLowerCase())
                        &&found.getType().toString().toLowerCase().contains(type)
                ){
                    this.device = found;
                }
            }
            if(!this.device.getType().toString().toLowerCase().contains("cpu")){
                this.device.setSharedMemory(false);// GPU's (!cpu's) don't share host memory!
            }
            System.out.println("\nChosen device:\n------------\n"+ device.toString()+"\n------------\n");
            System.out.println("\ndevice f_id:\n------------\n"+ this.device.getType().toString()+"\n------------\n");
            kernel = new TKernel();
            System.out.println("TDevice f kernel:\n------------");
            System.out.println(kernel.getTargetDevice().toString());
        }
    }

    public void dispose(){
        this.kernel.dispose();
    }

    public TKernel getKernel(){
        //System.out.println(this.kernel.cleanUpArrays());
        return kernel;
    }

    public boolean has(T tensor){
        return this.map.containsKey(tensor);
    }
    /**
     *  ======================================================================
     * */
    public void get(T tensor){
        if(kernel!=null){
            kernel.execute(
                    device.createRange(
                        kernel.executionSizeOf_fetchTsr(
                                register[0][
                                        map.get(
                                                tensor)], false
                        )
                    )
            );
            T.factory.inject(kernel.value(), false, tensor);
            if(tensor.rqsGradient()){
                kernel.execute(
                        device.createRange(
                                kernel.executionSizeOf_fetchTsr(
                                register[0][map.get(tensor)], true
                        ))
                );
                T.factory.inject(kernel.value(), true, tensor);
            }
            rmv(tensor);
        }
    }
    public void rmv(T tensor){
        if(kernel!=null) {
            if (map.containsKey(tensor)) {
                kernel.freePtrOf(map.get(tensor), register);
                map.remove(tensor);
                tensor.setIsOutsourced(false);
            }
        }
    }

    public TDevice add(T tensor){
        if(!map.containsKey(tensor)){
            map.put(tensor,
                    kernel.allocPtrFor(tensor, register)
            );
        }
        //for(int n : device.getMaxWorkItemSize()){
        //    System.out.print(n+", ");
        //}
        //System.out.println(device.getMaxWorkGroupSize()+"   <====  MAX WORK GROUP SIZE");
        kernel.execute(
                device.createRange(
                    kernel.executionSizeOf_storeTsr(register[0][map.get(tensor)], tensor.value(), false)
            )
        );
        if(tensor.rqsGradient()){
            double[] grd = (tensor.gradient()==null)?new double[tensor.value().length]:tensor.gradient();
            kernel.execute(
                device.createRange(
                        kernel.executionSizeOf_storeTsr(
                        register[0][map.get(tensor)],
                        grd, true
                    )
                )
            );
        }
        tensor.add(this);
        tensor.setIsOutsourced(true);
        return this;
    }

    public double[] valueOf(T tensor, boolean grd){
        kernel.execute(
                device.createRange(
                        kernel.executionSizeOf_fetchTsr(register[0][map.get(tensor)],grd)
                )
        );
        return kernel.value();
    }

    public void calculate(T[] tsrs, int f_id, int d){
        if(kernel==null){
            //What then?
        }else{
            int[] mode = new int[tsrs.length+2];
            mode[0] = f_id;
            mode[mode.length-1] = d;
            for(int mi=0; mi<(tsrs.length); mi++){
                mode[mi+1] = (tsrs[mi]!=null)?register[0][map.get(tsrs[mi])]:-1;
            }
            kernel.execute(
                device.createRange(
                    kernel.executionSizeOf_calc(mode)
                )
            );
        }

    }



    public void calculate_on_CPU(T drn, T t1, T t2, int f_id, int d){
        System.out.println("TEST STARTS:");
        System.out.println(stringified(kernel.values()));
        System.out.println(stringified(kernel.pointers()));
        System.out.println(stringified(kernel.shapes()));
        System.out.println(stringified(kernel.translations()));
        int[] m = new int[]{1, 2, 3};
        if(f_id<7){
            m = new int[]{f_id, register[0][map.get(drn)], register[0][map.get(t1)], (t2!=null)?register[0][map.get(t2)]:d};
        } else if(f_id<12){
            m = new int[]{f_id, register[0][map.get(drn)], register[0][map.get(t1)], d};
        } else {
            m = new int[]{f_id, register[0][map.get(drn)], register[0][map.get(t1)], register[0][map.get(t2)], d};
        }
        int size = kernel.executionSizeOf_calc(m);
        System.out.println("size: "+size);
        //kernel._mde = m;
        for(int i=0; i<size; i++){
            kernel.run(i, m);
            kernel._idx = new int[kernel._idx.length];
        }
        printDeviceContent(false);
    }

    public void printDeviceContent(boolean fetch){
        if(fetch){
            System.out.println(stringified(kernel.values()));
            System.out.println(stringified(kernel.pointers()));
            System.out.println(stringified(kernel.shapes()));
            System.out.println(stringified(kernel.translations()));
            System.out.println(stringified(kernel.idx()));
        }else{
            System.out.println(stringified(kernel._values));
            System.out.println(stringified(kernel._pointers));
            System.out.println(stringified(kernel._shapes));
            System.out.println(stringified(kernel._translations));
            System.out.println(stringified(kernel._idx));

        }

    }


    public String stringified(double[] a){
        String result = "";
        for(double ai : a) {
            result += ai+", ";
        }
        return result;
    }
    public String stringified(int[] a){
        String result = "";
        for(int ai : a) {
            result += ai+", ";
        }
        return result;
    }

}

