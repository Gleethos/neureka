package util;

import neureka.Tsr;
import neureka.acceleration.CPU;
import neureka.acceleration.Device;

import java.util.List;

public class NTester_Tensor extends NTester 
{
    public NTester_Tensor(String name)
    {
        super(name);
    }

    public int testShareDevice(Device device, Tsr[] tsrs){
        printSessionStart("Testing if tensors share device!");
        println(BAR +"  Device: "+device.toString());
        println(BAR +"-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+");
        for(Tsr t : tsrs){
            Device found = ((Device)t.find(Device.class));
            this.assertStringContains("result", (found==null)?"null":found.toString(), device.toString());
        }
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensor(Tsr tensor, List<String> expected){
        Object[] array = expected.toArray();
        String[] strings = new String[expected.size()];
        for(int i=0; i<strings.length; i++){
            strings[i] = (String)array[i];
        }
        return testTensor(tensor, strings);
    }

    public int testTensor(Tsr tensor, String[] expected){
        printSessionStart("Testing Tensor!");
        println(BAR +"  Tensor: "+tensor.toString());
        println(BAR +"-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+");
        String result = tensor.toString();
        for(String element : expected){
            this.assertStringContains("result", result, element);
        }
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensorAutoGrad(Tsr[] source, String operation, String[] expected){
        printSessionStart("Testing Tsr: autograd!");
        println(BAR +"  Function: "+operation);
        println(BAR +"-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+");
        Tsr product = new Tsr(source, operation);
        String result = product.toString("rc");
        for(String element : expected){
            this.assertStringContains("result", result, element);
        }
        product.delete();
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensorAutoGrad(Tsr[] source, String operation, String[] expected, Tsr error, double[][] expectedGradient){
        printSessionStart("Testing Tsr: autograd!");
        println(BAR +"  Function: "+operation);
        println(BAR +"-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+");
        Tsr product = new Tsr(source, operation);
        product.backward(error);
        String result = product.toString("rc");
        for(String element : expected){
            this.assertStringContains("result", result, element);
        }
        for(int i=0; i<source.length; i++){
            this.assertIsEqual(stringified(source[i].gradient64()), stringified(expectedGradient[i]));
        }
        product.delete();
        return (printSessionEnd()>0)?1:0;
    }


    public int testTensorUtility_reshape(int[] dim, int[] newForm, int[] expected){
        int[] result = Tsr.Util.Indexing.rearrange(dim, newForm);
        printSessionStart("Testing Tsr.indexing: dimension reshaping!");
        assertIsEqual(stringified(result), stringified(expected));
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensorUtility_translation(int[] dim, int[] expected){
        int [] result =  Tsr.Util.Indexing.newTlnOf(dim);
        printSessionStart("Testing Tsr.indexing: dimension _translation!");
        assertIsEqual(stringified(result), stringified(expected));
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensCon(int[] frstShp, int[] scndShp, double[] frstData, double[] scondData, double[] expctd){
        printSessionStart("Test Tsr.indexing: tensMul_mxd");
        int[] drnMxd  = Tsr.Util.Indexing.shpOfCon(frstShp, scndShp);
        double[] rsltData = new double[Tsr.Util.Indexing.szeOfShp(drnMxd)];
        CPU.exec.convolve_multiply(
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
        printSessionStart("Test Tsr.indexing: tensMul_mxd");
        int[] drnMxd  = Tsr.Util.Indexing.shpOfCon(frstShp, scndShp);
        CPU.exec.convolve_multiply_inverse(//inv
                new Tsr(frstShp, frstData),
                (first)?new Tsr(scndShp, scondData):new Tsr(drnMxd, drnData),
                (first)?new Tsr(drnMxd, drnData):new Tsr(scndShp, scondData)
        );
        assertIsEqual(stringified((first)?frstData:scondData), stringified(expctd));
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensBroadcast(int[] frstShp, int[] scndShp, double[] frstData, double[] scondData, double[] expctd){
        printSessionStart("Test Tsr.indexing: tensor broadcast_template.cl");
        int[] drnMxd  = Tsr.Util.Indexing.shpOfBrc(frstShp, scndShp);
        double[] rsltData = new double[Tsr.Util.Indexing.szeOfShp(drnMxd)];
        CPU.exec.broadcast_multiply(
                new Tsr(drnMxd, rsltData),
                new Tsr(frstShp, frstData),
                new Tsr(scndShp, scondData),
                -1
        );
        assertIsEqual(stringified(rsltData), stringified(expctd));
        return (printSessionEnd()>0)?1:0;
    }

    public int testInvTensBroadcast(
            int[] frstShp, int[] scndShp,
            double[] frstData, double[] scondData, double[] drnData,
            double[] expctd, boolean first
    ){
        printSessionStart("Test Tsr.indexing: tensor broadcast_template.cl");
        int[] drnMxd  = Tsr.Util.Indexing.shpOfBrc(frstShp, scndShp);
        CPU.exec.broadcast_multiply_inverse(//inv
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
                String str = result[i].toString("rdgc");
                for(String part : parts){
                    this.assertStringContains("tensor["+i+"]", str, part);
                }
            }
        }
        return (printSessionEnd()>0)?1:0;
    }


}
