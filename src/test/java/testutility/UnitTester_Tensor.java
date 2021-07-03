package testutility;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.GraphNode;
import neureka.backend.api.ExecutionCall;
import neureka.backend.standard.algorithms.Broadcast;
import neureka.backend.standard.algorithms.Convolution;
import neureka.devices.Device;
import neureka.devices.host.HostCPU;
import neureka.devices.opencl.OpenCLDevice;
import neureka.dtype.DataType;
import neureka.ndim.config.NDConfiguration;

import java.util.List;

public class UnitTester_Tensor extends UnitTester
{
    public UnitTester_Tensor(String name)
    {
        super(name);
    }

    public int testShareDevice( Device device, Tsr[] tsrs ){
        printSessionStart("Testing if tensors share device!");
        println(BAR +"  Device: "+device.toString());
        println(BAR +"-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+");
        for(Tsr t : tsrs){
            Device found = t.getDevice();//((Device)t.find(Device.class));
            this.assertStringContains("result", (found==null)?"null":found.toString(), device.toString());
        }
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensor(
            Tsr tensor,
            List<String> expected
    ){
        Object[] array = expected.toArray();
        String[] strings = new String[expected.size()];
        for(int i=0; i<strings.length; i++){
            strings[ i ] = (String)array[ i ];
        }
        DataType type = tensor.getDataType();
        if ( type != null ) {
            if ( tensor.getDevice() instanceof OpenCLDevice ) {
                this.assertStringContains("Tensor data type :", type.getTypeClass().getName(), "F32");
            } else this.assertStringContains("Tensor data type :", type.getTypeClass().getName(), "F64");
        }
        else this.assertStringContains("Tensor data type :", "null", "F64");
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
        Tsr product = Tsr.of(operation, source);
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
        Tsr product = Tsr.of(operation, source);
        product.backward(error);
        String result = product.toString("rc");
        for(String element : expected){
            this.assertStringContains("result", result, element);
        }
        for(int i=0; i<source.length; i++){
            this.assertIsEqual(stringified(source[ i ].gradient64()), stringified(expectedGradient[ i ]));
        }
        product.delete();
        return (printSessionEnd()>0)?1:0;
    }


    public int testTensorUtility_reshape(int[] dim, int[] newForm, int[] expected){
        int[] result = NDConfiguration.Utility.rearrange(dim, newForm);
        printSessionStart("Testing Tsr.indexing: dimension reshaping!");
        assertIsEqual(stringified(result), stringified(expected));
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensorUtility_translation(int[] dim, int[] expected){
        int [] result =  NDConfiguration.Utility.newTlnOf(dim);
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

    public int testTensCon(
            int[] frstShp, int[] scndShp, double[] frstData, double[] scondData, double[] expctd
    ){
        printSessionStart("Test tensor indexing: tensMul_mxd");
        int[] drnMxd  = Tsr.Utility.Indexing.shpOfCon(frstShp, scndShp);
        double[] rsltData = new double[NDConfiguration.Utility.szeOfShp(drnMxd)];
        Neureka.get().context().instance("x")
                .getAlgorithm(Convolution.class)
                .getImplementationFor( HostCPU.class )
                .run(
                        ExecutionCall.builder()
                                .device(HostCPU.instance())
                                .tensors(
                                        new Tsr[] {
                                                Tsr.of(drnMxd, rsltData),
                                                Tsr.of(frstShp, frstData),
                                                Tsr.of(scndShp, scondData)
                                        }
                                )
                                .derivativeIndex(-1)
                                .operation(Neureka.get().context().instance("x"))
                                .build()
                                .forDeviceType(HostCPU.class)
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
        Neureka.get().context().instance(((char) 171)+"x")
                .getAlgorithm(Convolution.class)
                .getImplementationFor( HostCPU.class )
                .run(
                        ExecutionCall.builder()
                            .device(HostCPU.instance())
                            .tensors(
                                    new Tsr[]{
                                        Tsr.of(frstShp, frstData),
                                        (first)?Tsr.of(scndShp, scondData):Tsr.of(drnMxd, drnData),
                                        (first)?Tsr.of(drnMxd, drnData):Tsr.of(scndShp, scondData)
                                    }
                            )
                            .derivativeIndex(0)
                            .operation(Neureka.get().context().instance(((char) 171)+"x"))
                            .build()
                            .forDeviceType(HostCPU.class)
                );
        assertIsEqual(stringified((first)?frstData:scondData), stringified(expctd));
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensBroadcast(int[] frstShp, int[] scndShp, double[] frstData, double[] scondData, double[] expctd){
        printSessionStart("Test Tsr.indexing: tensor broadcast_template.cl");
        int[] drnMxd  = Tsr.Utility.Indexing.shpOfBrc(frstShp, scndShp);
        double[] rsltData = new double[NDConfiguration.Utility.szeOfShp(drnMxd)];

        Neureka.get().context().instance("*")
                .getAlgorithm(Broadcast.class)
                .getImplementationFor( HostCPU.class )
                .run(
                        ExecutionCall.builder()
                            .device(HostCPU.instance())
                            .tensors(
                                    new Tsr[]{
                                            Tsr.of(drnMxd, rsltData),
                                            Tsr.of(frstShp, frstData),
                                            Tsr.of(scndShp, scondData)
                                    }
                            )
                            .derivativeIndex(-1)
                            .operation(Neureka.get().context().instance("*"))
                            .build()
                            .forDeviceType(HostCPU.class)
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
        Neureka.get().context().instance(((char) 171) + "*")
                .getAlgorithm(Broadcast.class)
                .getImplementationFor( HostCPU.class )
                .run(
                        ExecutionCall.builder()
                            .device(HostCPU.instance())
                            .tensors(
                                    new Tsr[]{
                                            Tsr.of(frstShp, frstData),
                                            (first)?Tsr.of(scndShp, scondData):Tsr.of(drnMxd, drnData),
                                            (first)?Tsr.of(drnMxd, drnData):Tsr.of(scndShp, scondData)
                                    }
                            )
                            .derivativeIndex(0)
                            .operation(Neureka.get().context().instance(((char) 171) + "*"))
                            .build()
                            .forDeviceType(HostCPU.class)
        );
        assertIsEqual(stringified((first)?frstData:scondData), stringified(expctd));
        Neureka.get().context().instance("*" + ((char) 187))
                .getAlgorithm(Broadcast.class)
                .getImplementationFor( HostCPU.class )
                .run(
                        ExecutionCall.builder()
                            .device(HostCPU.instance())
                            .tensors(
                                    new Tsr[]{
                                            Tsr.of(frstShp, frstData),
                                            (first)?Tsr.of(scndShp, scondData):Tsr.of(drnMxd, drnData),
                                            (first)?Tsr.of(drnMxd, drnData):Tsr.of(scndShp, scondData),
                                    }
                            )
                            .derivativeIndex(0)
                            .operation(Neureka.get().context().instance("*" + ((char) 187)))
                            .build()
                            .forDeviceType(HostCPU.class)
        );
        assertIsEqual(stringified((first)?frstData:scondData), stringified(expctd));

        return ( printSessionEnd() > 0 ) ? 1 : 0;
    }


    /**
     * @param tensors
     * @param f
     * @return
     */
    public int testInjection( Tsr[] tensors, String f, String[][] expected )
    {
        printSessionStart("Test injection: I[0] <- I[1], I[0] -> I[1] : "+f);
        Tsr[] result = new Tsr[tensors.length+1];
        result[0] = Tsr.of(tensors, f, false);
        assert result[0].find( GraphNode.class ) == null; // Because "doAD" is false!
        System.arraycopy(tensors, 0, result, 1, result.length - 1);
        for(int i=0; i<result.length; i++){
            if(expected[ i ]!=null && expected[ i ].length!=0){
                String[] parts = expected[ i ];
                String str = result[ i ].toString("rdgc");
                for(String part : parts){
                    this.assertStringContains("tensor["+i+"]", str, part);
                }
            }
        }
        return (printSessionEnd()>0)?1:0;
    }




}
