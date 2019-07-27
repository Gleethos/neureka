package neureka.unit.state;

import neureka.core.T;
import neureka.core.module.calc.Device;
import neureka.core.module.calc.dcomp.TKernel;
import neureka.unit.NVTesting;
import neureka.utility.NMessageFrame;

public class NVTesting_TensorDevice extends NVTesting {

    public NVTesting_TensorDevice(NMessageFrame console, NMessageFrame resultConsole)
    {
        super(console, resultConsole);
    }

    public int testAddTensor(Device device, T tensor, double[] values, int[] shapes, int[] translations, int[] pointers){

        double[] value = tensor.value();
        int[] shape = tensor.shape();
        int[] translation = tensor.translation();

        this.printSessionStart("Adding tensor to device");
        this.assertEqual("tensor.isOutsourced()", ""+tensor.isOutsourced(), "false");
        device.add(tensor);
        this.assertEqual("tensor.isOutsourced()", ""+tensor.isOutsourced(), "true");
        TKernel kernel = device.getKernel();
        this.assertEqual("kernel.values()", stringified(kernel.values()), stringified(values));
        this.assertEqual("kernel.shapes()", stringified(kernel.shapes()), stringified(shapes));
        this.assertEqual("kernel.translations()", stringified(kernel.translations()), stringified(translations));

        this.assertEqual("tensor.value()", stringified(tensor.value()),stringified(value));
        this.assertEqual("kernel.value()", stringified(kernel.value()),stringified(value));
        this.assertEqual("kernel.shape()", stringified(kernel.shape()),stringified(shape));
        this.assertEqual("kernel.translation()", stringified(kernel.translation()),stringified(translation));

        this.assertEqual("kernel.pointers()", stringified(kernel.pointers()), stringified(pointers));

        return this.printSessionEnd();
    }

    public int testGetTensor(Device device, T tensor, double[] values, int[] shapes, int[] translations, int[] pointers){

        double[] value = tensor.value();
        int[] shape = tensor.shape();
        int[] translation = tensor.translation();


        this.printSessionStart("Getting tensor from Device");
        TKernel kernel = device.getKernel();
        this.assertEqual("tensor.isOutsourced()", ""+tensor.isOutsourced(), "true");
        device.get(tensor);
        this.assertEqual("tensor.isOutsourced()", ""+tensor.isOutsourced(), "false");
        this.assertEqual("kernel.values()", stringified(kernel.values()), stringified(values));
        this.assertEqual("kernel.shapes()", stringified(kernel.shapes()), stringified(shapes));
        this.assertEqual("kernel.translations()", stringified(kernel.translations()), stringified(translations));


        this.assertEqual("kernel.values()", stringified(kernel.values()), stringified(values));
        this.assertEqual("kernel.shapes()", stringified(kernel.shapes()), stringified(shapes));
        this.assertEqual("kernel.translations()", stringified(kernel.translations()), stringified(translations));

        this.assertEqual("kernel.pointers()", stringified(kernel.pointers()), stringified(pointers));

        return this.printSessionEnd();
    }

    public int testCalculation(Device device, T drn, T src1, T src2, int f_id, double[] values){

        String message = "";
        message = (f_id==18)?"Tensor product":message;
        message = (f_id==17)?"Tensor addition":message;
        message = (f_id==16)?"Tensor subtracting":message;
        message = (f_id==15)?"Tensor modulo":message;
        message = (f_id==14)?"Tensor multiplication":message;
        message = (f_id==13)?"Tensor division":message;
        message = (f_id==12)?"Tensor power":message;
        message = (f_id==11)?"Tensor cosinus":message;
        message = (f_id==10)?"Tensor sinus":message;
        message = (f_id==9)?"Tensor absolute":message;
        message = (f_id==8)?"Tensor pi (product summation)":message;
        message = (f_id==7)?"Tensor summation":message;
        message = (f_id==6)?"Tensor gaussian":message;
        message = (f_id==5)?"Tensor linear":message;
        message = (f_id==4)?"Tensor softplus":message;
        message = (f_id==3)?"Tensor quadratic":message;
        message = (f_id==2)?"Tensor tanh":message;
        message = (f_id==1)?"Tensor sigmoid":message;
        message = (f_id==5)?"Tensor relu":message;

        this.printSessionStart(message);

        //for(int i=0; i<50000; i++){
            //device.calculate(drn, src1, src2, f_id);
        if(src2==null){
            device.calculate(new T[]{drn, src1}, f_id, -1);
        }else{
            device.calculate(new T[]{drn, src1, src2}, f_id, -1);
        }
        //}



        this.assertEqual("kernel.values()", stringified(device.getKernel().values()), stringified(values));

        return this.printSessionEnd();
    }




}
