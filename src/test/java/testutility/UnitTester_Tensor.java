package testutility;

import neureka.Tsr;
import neureka.acceleration.host.HostCPU;
import neureka.acceleration.Device;
import neureka.acceleration.host.execution.HostExecutor;
import neureka.calculus.backend.ExecutionCall;
import neureka.calculus.backend.operations.OperationType;
import neureka.calculus.backend.implementations.functional.Broadcast;
import neureka.calculus.backend.implementations.functional.Convolution;

import java.util.List;

public class UnitTester_Tensor extends UnitTester
{
    public UnitTester_Tensor(String name)
    {
        super(name);
    }

    public int testShareDevice(Device device, Tsr[] tsrs){
        printSessionStart("Testing if tensors share device!");
        println(BAR +"  Device: "+device.toString());
        println(BAR +"-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+");
        for(Tsr t : tsrs){
            Device found = t.device();//((Device)t.find(Device.class));
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
        String result = tensor.toString();
        println(BAR +"  Tensor: "+result);
        println(BAR +"-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+");
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
        int[] result = Tsr.Utility.Indexing.rearrange(dim, newForm);
        printSessionStart("Testing Tsr.indexing: dimension reshaping!");
        assertIsEqual(stringified(result), stringified(expected));
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensorUtility_translation(int[] dim, int[] expected){
        int [] result =  Tsr.Utility.Indexing.newTlnOf(dim);
        printSessionStart("Testing Tsr.indexing: dimension _translation!");
        assertIsEqual(stringified(result), stringified(expected));
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensorUtility_makeFit(int[] a, int[] b, int[][] expected){
        int [][] result =  Tsr.Utility.Indexing.makeFit(a, b);
        printSessionStart("Testing Tsr.indexing: dimension _translation!");
        assertTrue("Invalid result!", result!=null && result.length==2);
        assertIsEqual(
                stringified(result[0]),
                stringified(expected[0])
        );
        assertIsEqual(
                stringified(result[1]),
                stringified(expected[1])
        );
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensCon(int[] frstShp, int[] scndShp, double[] frstData, double[] scondData, double[] expctd){
        printSessionStart("Test Tsr.indexing: tensMul_mxd");
        int[] drnMxd  = Tsr.Utility.Indexing.shpOfCon(frstShp, scndShp);
        double[] rsltData = new double[Tsr.Utility.Indexing.szeOfShp(drnMxd)];
        OperationType.instance("x")
                .getImplementation(Convolution.class)
                .getExecutor(HostExecutor.class)
                .getExecution().run(
                new ExecutionCall<>(
                        HostCPU.instance(),
                        new Tsr[]{
                                new Tsr(drnMxd, rsltData),
                                new Tsr(frstShp, frstData),
                                new Tsr(scndShp, scondData)
                        }, -1,
                        OperationType.instance("x")
                )
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
        int[] drnMxd  = Tsr.Utility.Indexing.shpOfCon(frstShp, scndShp);
        OperationType.instance(((char) 171)+"x")
                .getImplementation(Convolution.class)
                .getExecutor(HostExecutor.class)
                .getExecution().run(
                new ExecutionCall<>(
                        HostCPU.instance(),
                        new Tsr[]{
                                new Tsr(frstShp, frstData),
                                (first)?new Tsr(scndShp, scondData):new Tsr(drnMxd, drnData),
                                (first)?new Tsr(drnMxd, drnData):new Tsr(scndShp, scondData)
                        },
                        0,
                        OperationType.instance(((char) 171)+"x")
                )
        );
        assertIsEqual(stringified((first)?frstData:scondData), stringified(expctd));
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensBroadcast(int[] frstShp, int[] scndShp, double[] frstData, double[] scondData, double[] expctd){
        printSessionStart("Test Tsr.indexing: tensor broadcast_template.cl");
        int[] drnMxd  = Tsr.Utility.Indexing.shpOfBrc(frstShp, scndShp);
        double[] rsltData = new double[Tsr.Utility.Indexing.szeOfShp(drnMxd)];

        OperationType.instance("*")
                .getImplementation(Broadcast.class)
                .getExecutor(HostExecutor.class)
                .getExecution().run(
                        new ExecutionCall<>(
                                HostCPU.instance(),
                                new Tsr[]{
                                        new Tsr(drnMxd, rsltData),
                                        new Tsr(frstShp, frstData),
                                        new Tsr(scndShp, scondData)
                                },
                                -1,
                                OperationType.instance("*")
                        )
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
        int[] drnMxd  = Tsr.Utility.Indexing.shpOfBrc(frstShp, scndShp);
        OperationType.instance(((char) 171) + "*")
                .getImplementation(Broadcast.class)
                .getExecutor(HostExecutor.class)
                .getExecution().run(
                new ExecutionCall<>(
                        HostCPU.instance(),
                        new Tsr[]{
                                new Tsr(frstShp, frstData),
                                (first)?new Tsr(scndShp, scondData):new Tsr(drnMxd, drnData),
                                (first)?new Tsr(drnMxd, drnData):new Tsr(scndShp, scondData)
                        },
                        0,
                        OperationType.instance(((char) 171) + "*")
                )
        );
        assertIsEqual(stringified((first)?frstData:scondData), stringified(expctd));
        OperationType.instance("*" + ((char) 187))
                .getImplementation(Broadcast.class)
                .getExecutor(HostExecutor.class)
                .getExecution().run(
                new ExecutionCall<>(
                        HostCPU.instance(),
                        new Tsr[]{
                                new Tsr(frstShp, frstData),
                                (first)?new Tsr(scndShp, scondData):new Tsr(drnMxd, drnData),
                                (first)?new Tsr(drnMxd, drnData):new Tsr(scndShp, scondData),
                        },
                        0,
                        OperationType.instance("*" + ((char) 187))
                )
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
        System.arraycopy(tensors, 0, result, 1, result.length - 1);
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
