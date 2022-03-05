package testutility;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.autograd.GraphNode;
import neureka.backend.api.ExecutionCall;
import neureka.backend.standard.algorithms.Broadcast;
import neureka.backend.standard.algorithms.Convolution;
import neureka.backend.standard.algorithms.internal.Fun;
import neureka.backend.standard.implementations.CPUImplementation;
import neureka.calculus.internal.CalcUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import neureka.dtype.DataType;
import neureka.ndim.config.NDConfiguration;
import org.jetbrains.annotations.Contract;

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
            Device found = t.getDevice();//((Device)t.get(Device.class));
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
        for ( String element : expected ) {
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
        boolean productInInputs = false;
        for ( Tsr t : source ) productInInputs = (t == product || productInInputs);
        if ( !productInInputs ) product.getUnsafe().delete();
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
            double[] gradient;
            if ( source[ i ].hasGradient() && source[ i ].is(Float.class) ) {
                float[] data = (float[]) source[i].getGradient().getValue();
                gradient = new double[data.length];
                for ( int ii = 0; ii < data.length; ii++ ) gradient[ii] = data[ii];
            } else
                gradient = source[ i ].hasGradient()
                                ? (double[])source[ i ].getGradient().getValue()
                                : null;
            this.assertIsEqual(
                    stringified(gradient),
                    stringified(expectedGradient[ i ])
            );
        }
        product.getUnsafe().delete();
        return (printSessionEnd()>0)?1:0;
    }


    public int testTensorUtility_reshape(int[] dim, int[] newForm, int[] expected){
        int[] result = NDConfiguration.Utility.rearrange(dim, newForm);
        printSessionStart("Testing Tsr.indexing: dimension reshaping!");
        assertIsEqual(stringified(result), stringified(expected));
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensorUtility_translation(int[] dim, int[] expected){
        int [] result =  NDConfiguration.Layout.ROW_MAJOR.newTranslationFor(dim);
        printSessionStart("Testing Tsr.indexing: dimension _translation!");
        assertIsEqual(stringified(result), stringified(expected));
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensorUtility_makeFit( int[] a, int[] b, int[][] expected ){
        int [][] result =  Tsr.Utility.makeFit( a, b );
        printSessionStart("Testing Tsr.indexing: dimension _translation!");
        if ( result.length != 2 ) throw new AssertionError("Invalid result!");
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
        int[] drnMxd  = _shpOfCon(frstShp, scndShp);
        double[] rsltData = new double[NDConfiguration.Utility.sizeOfShape(drnMxd)];
        Neureka.get().backend().getOperation("x")
                .getAlgorithm(Convolution.class)
                .getImplementationFor( CPU.class )
                .run(
                        ExecutionCall.of(
                                    Tsr.of(drnMxd, rsltData),
                                    Tsr.of(frstShp, frstData),
                                    Tsr.of(scndShp, scondData)
                                )
                                .andArgs(Arg.DerivIdx.of(-1))
                                .running(Neureka.get().backend().getOperation("x"))
                                .on(CPU.get())
                                .forDeviceType(CPU.class)
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
        int[] drnMxd  = _shpOfCon(frstShp, scndShp);
        Neureka.get().backend().getOperation(((char) 171)+"x")
                .getAlgorithm(Convolution.class)
                .getImplementationFor( CPU.class )
                .run(
                        ExecutionCall.of(
                                Tsr.of(frstShp, frstData),
                                (first)?Tsr.of(scndShp, scondData):Tsr.of(drnMxd, drnData),
                                (first)?Tsr.of(drnMxd, drnData):Tsr.of(scndShp, scondData)
                            )
                            .andArgs( Arg.DerivIdx.of(0) )
                            .running(Neureka.get().backend().getOperation(((char) 171)+"x"))
                            .on(CPU.get())
                            .forDeviceType(CPU.class)
                );
        assertIsEqual(stringified((first)?frstData:scondData), stringified(expctd));
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensBroadcast(int[] frstShp, int[] scndShp, double[] frstData, double[] scondData, double[] expctd){
        printSessionStart("Test Tsr.indexing: tensor broadcast_template.cl");
        int[] drnMxd  = _shpOfBrc(frstShp, scndShp);
        double[] rsltData = new double[NDConfiguration.Utility.sizeOfShape(drnMxd)];

        Neureka.get().backend().getOperation("*")
                .getAlgorithm(Broadcast.class)
                .getImplementationFor( CPU.class )
                .run(
                        ExecutionCall.of(
                                Tsr.of(drnMxd, rsltData),
                                Tsr.of(frstShp, frstData),
                                Tsr.of(scndShp, scondData)
                            )
                            .andArgs(Arg.DerivIdx.of(-1))
                            .running(Neureka.get().backend().getOperation("*"))
                            .on(CPU.get())
                            .forDeviceType(CPU.class)
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
        int[] drnMxd  = _shpOfBrc(frstShp, scndShp);

        Broadcast right = new Broadcast((executionCall, executor) -> null)
                                .setCanPerformBackwardADFor( call -> true )
                                .setCanPerformForwardADFor(
                                        call -> {
                                            Tsr<?> last = null;
                                            for ( Tsr<?> t : call.inputs() ) {
                                                if ( last != null && !last.shape().equals(t.shape()) ) return false;
                                                last = t; // Note: shapes are cached!
                                            }
                                            return true;
                                        }
                                )
                                .setSupplyADAgentFor(
                                        (Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                                        {
                                            Tsr<?> ctxDerivative = (Tsr<?>) call.getValOf(Arg.Derivative.class);
                                            Function mul = Neureka.get().backend().getFunction().mul();
                                            if ( ctxDerivative != null ) {
                                                return ADAgent.of( ctxDerivative )
                                                        .setForward( (node, forwardDerivative ) -> mul.execute( forwardDerivative, ctxDerivative ) )
                                                        .setBackward( (node, forwardDerivative ) -> mul.execute( forwardDerivative, ctxDerivative ) );
                                            }
                                            int d = call.getDerivativeIndex();
                                            if ( forward ) throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                                            else
                                            {
                                                Tsr<?> derivative = f.executeDerive( call.inputs(), d );
                                                return ADAgent.of( derivative )
                                                        .setForward( (node, forwardDerivative ) -> mul.execute( forwardDerivative, derivative ) )
                                                        .setBackward( (node, backwardError ) -> mul.execute( backwardError, derivative ) );
                                            }
                                        }
                                )
                                .setExecutionDispatcher( CalcUtil::defaultRecursiveExecution)
                                .setCallPreparation(
                                        call -> {
                                            int offset = ( call.input( 0 ) == null ) ? 1 : 0;
                                            return
                                                ExecutionCall.of(call.input( offset ), call.input( 1+offset )).andArgs(Arg.DerivIdx.of(-1)).running(Neureka.get().backend().getOperation("idy")).on( call.getDevice() );
                                        }
                                )
                                .buildFunAlgorithm()
                                .setImplementationFor(
                                        CPU.class,
                                        CPUImplementation
                                                .withArity(3)
                                                .andImplementation(
                                                        Broadcast.implementationForCPU()
                                                                .with(Fun.F64F64ToF64.triple(
                                                                        ( a, b ) -> a * b,
                                                                        ( a, b ) -> a * b,
                                                                        ( a, b ) -> a * b
                                                                ))
                                                                .with(Fun.F32F32ToF32.triple(
                                                                        ( a, b ) -> a * b,
                                                                        ( a, b ) -> a * b,
                                                                        ( a, b ) -> a * b
                                                                ))
                                                                .get()
                                                )
                                );

        Broadcast left = new Broadcast((executionCall, executor) -> null)
                                    .setCanPerformBackwardADFor( call -> true )
                                    .setCanPerformForwardADFor(
                                            call -> {
                                                Tsr<?> last = null;
                                                for ( Tsr<?> t : call.inputs() ) {
                                                    if ( last != null && !last.shape().equals(t.shape()) ) return false;
                                                    last = t; // Note: shapes are cached!
                                                }
                                                return true;
                                            }
                                    )
                                    .setSupplyADAgentFor(
                                            (Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                                            {
                                                Tsr<?> ctxDerivative = (Tsr<?>) call.getValOf(Arg.Derivative.class);
                                                Function mul = Neureka.get().backend().getFunction().mul();
                                                if ( ctxDerivative != null ) {
                                                    return ADAgent.of( ctxDerivative )
                                                            .setForward( (node, forwardDerivative ) -> mul.execute( forwardDerivative, ctxDerivative ) )
                                                            .setBackward( null );
                                                }
                                                Tsr<?>[] inputs = call.inputs();
                                                int d = call.getDerivativeIndex();
                                                if ( forward ) throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                                                else
                                                {
                                                    Tsr<?> derivative = f.executeDerive( inputs, d );
                                                    return ADAgent.of( derivative )
                                                            .setForward( ( node, forwardDerivative ) -> mul.execute( forwardDerivative, derivative ) )
                                                            .setBackward( ( node, backwardError ) -> mul.execute( backwardError, derivative ) );
                                                }
                                            }
                                    )
                                    .setExecutionDispatcher( CalcUtil::defaultRecursiveExecution)
                                    .setCallPreparation(
                                            call -> {
                                                Tsr<?>[] tsrs = call.inputs();
                                                int offset = ( tsrs[ 0 ] == null ) ? 1 : 0;
                                                return ExecutionCall.of(tsrs[offset], tsrs[1+offset]).andArgs(Arg.DerivIdx.of(-1)).running(Neureka.get().backend().getOperation("idy")).on( call.getDevice() );
                                            }
                                    )
                                    .buildFunAlgorithm()
                                    .setImplementationFor(
                                            CPU.class,
                                            CPUImplementation
                                                    .withArity(3)
                                                    .andImplementation(
                                                            Broadcast.implementationForCPU()
                                                                    .with(Fun.F64F64ToF64.triple(
                                                                            ( a, b ) -> a * b,
                                                                            ( a, b ) -> a * b,
                                                                            ( a, b ) -> a * b
                                                                    ))
                                                                    .with(Fun.F32F32ToF32.triple(
                                                                            ( a, b ) -> a * b,
                                                                            ( a, b ) -> a * b,
                                                                            ( a, b ) -> a * b
                                                                    ))
                                                                    .get()
                                                    )
                                    );

        left.getImplementationFor( CPU.class )
                .run(
                        ExecutionCall.of(
                                    Tsr.of(frstShp, frstData),
                                    (first)?Tsr.of(scndShp, scondData):Tsr.of(drnMxd, drnData),
                                    (first)?Tsr.of(drnMxd, drnData):Tsr.of(scndShp, scondData)
                            )
                            .andArgs( Arg.DerivIdx.of(0) )
                            .running(Neureka.get().backend().getOperation(((char) 171) + "*"))
                            .on(CPU.get())
                            .forDeviceType(CPU.class)
                );
        assertIsEqual(stringified((first)?frstData:scondData), stringified(expctd));

        right.getImplementationFor( CPU.class )
                .run(
                        ExecutionCall.of(
                                    Tsr.of(frstShp, frstData),
                                    (first)?Tsr.of(scndShp, scondData):Tsr.of(drnMxd, drnData),
                                    (first)?Tsr.of(drnMxd, drnData):Tsr.of(scndShp, scondData)
                            )
                            .andArgs( Arg.DerivIdx.of(0) )
                            .running(Neureka.get().backend().getOperation("*" + ((char) 187)))
                            .on(CPU.get())
                            .forDeviceType(CPU.class)
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
        result[0] = Tsr.of( f, false, tensors );
        assert result[0] == null || result[0].get( GraphNode.class ) == null; // Because "doAD" is false!
        System.arraycopy(tensors, 0, result, 1, result.length - 1);
        for(int i=0; i<result.length; i++){
            if(expected[ i ]!=null && expected[ i ].length!=0){
                String[] parts = expected[ i ];
                String str = ( result[ i ] == null ? "null" : result[ i ].toString("rdgc") );
                for(String part : parts){
                    this.assertStringContains("tensor["+i+"]", str, part);
                }
            }
        }
        return (printSessionEnd()>0)?1:0;
    }

    @Contract(pure = true)
    private static int[] _shpOfBrc( int[] shp1, int[] shp2 ) {
        int[] shape = new int[ ( shp1.length + shp2.length ) / 2 ];
        for ( int i = 0; i < shp1.length && i < shp2.length; i++ ) {
            shape[ i ] = Math.max( shp1[ i ], shp2[ i ] );
            if ( Math.min(shp1[ i ], shp2[ i ]) != 1 && Math.max( shp1[ i ], shp2[ i ] ) != shape[ i ] ) {
                throw new IllegalStateException("Broadcast not possible. Shapes do not match!");
            }
        }
        return shape;
    }

    @Contract(pure = true)
    private static int[] _shpOfCon( int[] shp1, int[] shp2 ) {
        int[] shape = new int[ ( shp1.length + shp2.length ) / 2 ];
        for ( int i = 0; i < shp1.length && i < shp2.length; i++ )
            shape[ i ] = Math.abs( shp1[ i ] - shp2[ i ] ) + 1;
        return shape;
    }

}
