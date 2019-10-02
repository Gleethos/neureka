package util;

import neureka.core.Tsr;
import neureka.core.device.Device;
import neureka.core.function.factory.Function;

public class NTester_Tensor extends NTester {

    public NTester_Tensor(String name)
    {
        super(name);
    }

    public int testShareDevice(Device device, Tsr[] tsrs){
        printSessionStart("Testing if tensors share device!");
        println(bar+"  Device: "+device.toString());
        println(bar+"-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+");
        for(Tsr t : tsrs){
            Device found = ((Device)t.find(Device.class));
            this.assertStringContains("result", (found==null)?"null":found.toString(), device.toString());
        }
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensor(Tsr tensor, String[] expected){
        printSessionStart("Testing Tensor!");
        println(bar+"  Tensor: "+tensor.toString());
        println(bar+"-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+");
        String result = tensor.toString();
        for(String element : expected){
            this.assertStringContains("result", result, element);
        }
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensorAutoGrad(Tsr[] source, String operation, String[] expected){
        printSessionStart("Testing Tsr: autograd!");
        println(bar+"  IFunction: "+operation);
        println(bar+"-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+");
        Tsr product = new Tsr(source, operation);
        String result = product.toString("r");
        for(String element : expected){
            this.assertStringContains("result", result, element);
        }
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensorAutoGrad(Tsr[] source, String operation, String[] expected, Tsr error, double[][] expectedGradient){
        printSessionStart("Testing Tsr: autograd!");
        println(bar+"  IFunction: "+operation);
        println(bar+"-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+");
        Tsr product = new Tsr(source, operation);
        product.backward(error);
        String result = product.toString("r");
        for(String element : expected){
            this.assertStringContains("result", result, element);
        }
        for(int i=0; i<source.length; i++){
            this.assertIsEqual(stringified(source[i].gradient()), stringified(expectedGradient[i]));
        }
        return (printSessionEnd()>0)?1:0;
    }


    public int testTensorUtility_reshape(int[] dim, int[] newForm, int[] expected){
        int[] result = Tsr.factory.util.rearrange(dim, newForm);
        printSessionStart("Testing Tsr.util: dimension reshaping!");
        assertIsEqual(stringified(result), stringified(expected));
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensorUtility_translation(int[] dim, int[] expected){
        int [] result =  Tsr.factory.util.idxTln(dim);
        printSessionStart("Testing Tsr.util: dimension _translation!");
        assertIsEqual(stringified(result), stringified(expected));
        return (printSessionEnd()>0)?1:0;
    }
    public int testTensorBase_idxFromAnchor(int[] dim, int idx, int[] expected){
        int [] result =  Tsr.factory.util.idxOf(idx, Tsr.factory.util.idxTln(dim));
        printSessionStart("Testing Tsr.util: _shape to _translation to index!");
        assertIsEqual(stringified(result), stringified(expected));
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensCon(int[] frstShp, int[] scndShp, double[] frstData, double[] scondData, double[] expctd){
        printSessionStart("Test Tsr.util: tensMul_mxd");
        int[] drnMxd  = Tsr.factory.util.shpOfCon(frstShp, scndShp);
        double[] rsltData = new double[Tsr.factory.util.szeOfShp(drnMxd)];
        Function.exec.convection(
                new Tsr(drnMxd, rsltData),
                new Tsr(frstShp, frstData),
                new Tsr(scndShp, scondData)
        );
        assertIsEqual(stringified(rsltData), stringified(expctd));
        return (printSessionEnd()>0)?1:0;
    }

    public int testInvTensCon(
            int[] frstShp, int[] scndShp,
            double[] frstData, double[] scondData, double[] drnData,
            double[] expctd, boolean first
    ){
        printSessionStart("Test Tsr.util: tensMul_mxd");
        int[] drnMxd  = Tsr.factory.util.shpOfCon(frstShp, scndShp);
        Function.exec.convection_inv(
                new Tsr(frstShp, frstData),
                (first)?new Tsr(scndShp, scondData):new Tsr(drnMxd, drnData),
                (first)?new Tsr(drnMxd, drnData):new Tsr(scndShp, scondData)
        );
        assertIsEqual(stringified((first)?frstData:scondData), stringified(expctd));
        return (printSessionEnd()>0)?1:0;
    }

    /**
     * @param tensors
     * @param f
     * @return
     */
    public int testInjection(Tsr[] tensors, String f, String[][] expected)
    {
        printSessionStart("Test injection: I[0] <- I[1], I[0] -> I[1] : "+f);
        Tsr[] result = new Tsr[tensors.length+1];
        result[0] = new Tsr(tensors, f);
        for(int i=1; i<result.length; i++){
            result[i] = tensors[i-1];
        }
        for(int i=0; i<result.length; i++){
            if(expected[i]!=null && expected[i].length!=0){
                String[] parts = expected[i];
                String str = result[i].toString("rdg");
                for(String part : parts){
                    this.assertStringContains("tensor["+i+"]", str, part);
                }
            }
        }
        return (printSessionEnd()>0)?1:0;
    }


}
