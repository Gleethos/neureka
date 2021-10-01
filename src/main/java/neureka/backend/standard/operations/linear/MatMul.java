package neureka.backend.standard.operations.linear;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.Algorithm;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Convolution;
import neureka.backend.standard.algorithms.FunAlgorithm;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.backend.standard.implementations.HostImplementation;
import neureka.calculus.CalcUtil;
import neureka.calculus.Function;
import neureka.calculus.RecursiveExecutor;
import neureka.calculus.args.Arg;
import neureka.calculus.assembly.FunctionBuilder;
import neureka.devices.Device;
import neureka.devices.host.HostCPU;
import neureka.devices.opencl.OpenCLDevice;
import neureka.ndim.config.types.simple.SimpleD2Configuration;

public class MatMul extends AbstractOperation
{

    public MatMul()
    {
        super(
                new OperationBuilder()
                        .setFunction(         "matmul"    )
                        .setOperator(         "@"         )
                        .setArity(            2           )
                        .setIsOperator(       true        )
                        .setIsIndexer(        false       )
                        .setIsDifferentiable( true        )
                        .setIsInline(         false       )
        );

        FunAlgorithm simpleMatMulAlgorithm =
                        Algorithm.withName("simple_matmul")
                                    .setIsSuitableFor(
                                            call -> call.validate()
                                                        .all( t -> t.getNDConf() instanceof SimpleD2Configuration )
                                                        .estimation()
                                    )
                                    .setCanPerformBackwardADFor( call -> true )
                                    .setCanPerformForwardADFor( call -> false )
                                    .setSupplyADAgentFor(
                                        ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                                        {
                                            if ( forward ) throw new IllegalArgumentException("Matrix multiplication does not support forward-AD!");

                                            Function invX = Neureka.get().context().getFunction().matMul();
                                            Tsr<?>[] inputs = call.getTensors();
                                            int d = (1 + call.getValOf( Arg.DerivIdx.class )) % 2;
                                            Tsr<?> deriv = inputs[ d ].T().clone();
                                            if ( d == 0 )
                                                return ADAgent.of( deriv )
                                                                .setBackward( (node, error) -> invX.execute( error, deriv ) );
                                            else
                                                return ADAgent.of( deriv )
                                                                .setBackward( (node, error) -> invX.execute( error, deriv ) );
                                        }
                                    )
                                    .setExecutionDispatcher(
                                        ( caller, call ) -> {
                                            if ( !caller.isFlat() )
                                                return CalcUtil.defaultRecursiveExecution( caller, call );

                                            Tsr<?>[] tensors = CalcUtil.srcActivation(call.getTensors(), call.getJ(), -1, 1, caller.getSubFunctions().toArray(new Function[0]));
                                            for ( Tsr<?> t : tensors ) if ( t != null ) t.setIsVirtual( false );
                                            ExecutionCall<HostCPU> preparedCall = (ExecutionCall<HostCPU>) call.getAlgorithm().prepare( call.withTensors(tensors) );
                                            return call.getAlgorithm()
                                                        .getImplementationFor(HostCPU.class)
                                                        .runAndGetFirstTensor(preparedCall);
                                        }
                                    )
                                    .setCallPreparation(
                                        call -> {
                                            Tsr<?>[] tsrs = call.getTensors();
                                            Device device = call.getDevice();
                                            if ( tsrs[ 0 ] == null ) // Creating a new tensor:
                                            {
                                                int[] shp = new int[]{ tsrs[ 1 ].shape(0), tsrs[ 2 ].shape(1) };
                                                Tsr<?> output = Tsr.of( shp, 0.0 );
                                                output.setIsVirtual( false );
                                                try {
                                                    device.store(output);
                                                } catch ( Exception e ) {
                                                    e.printStackTrace();
                                                }
                                                tsrs[ 0 ] = output;
                                            }
                                            return call;
                                        }
                                    )
                                    .buildFunAlgorithm();

        setAlgorithm(
                simpleMatMulAlgorithm
                        .setImplementationFor(
                                HostCPU.class,
                                new HostImplementation(
                                        call ->
                                        {
                                            double[] A = (double[]) call.getTsrOfType(Double.class, 1).getData();
                                            double[] B = (double[]) call.getTsrOfType(Double.class, 2).getData();
                                            double[] C = (double[]) call.getTsrOfType(Double.class, 0).getData();

                                            int[] shapeA = call.getTsrOfType(Double.class, 1).getNDConf().shape();
                                            int[] shapeB = call.getTsrOfType(Double.class, 2).getNDConf().shape();
                                            int[] shapeC = call.getTsrOfType(Double.class, 0).getNDConf().shape();

                                            SimpleMatMul.execute(A, shapeA, B, shapeB, C, shapeC);
                                        },
                                        3
                                )
                        )
                        .setImplementationFor(
                                OpenCLDevice.class,
                                CLImplementation.fromSource()
                                        .arity( 3 )
                                        .kernelName( "simpleMatMul" )
                                        .kernelSource(
                                                "_kernel void simpleMatMul(   " +
                                                        "   int widthA,                                     " +
                                                        "   int heightA,                                    " +
                                                        "   int widthB,                                     " +
                                                        "   int heightB,                                    " +
                                                        "   __global float* outputC,                        " +
                                                        "   __global float* inputA,                         " +
                                                        "   __global float* inputB                          " +
                                                        ") {                                                " +
                                                        "   int row = get_global_id( 1 );                     " +
                                                        "   int col = get_global_id(0);                     " +
                                                        "   float sum = 0.0f;                               " +
                                                        "   for ( int i = 0; i < widthA; i++ ) {            " +
                                                        "      sum += inputA[ row * widthA + i ] * inputB[ i * widthB + col ];" +
                                                        "   }                                               " +
                                                        "   outputC[ row * widthB * col ] = sum;" +
                                                        "}"
                                        )
                                        .build()
                        )
        );
        // TODO: Non-simple tensors:
        CLImplementation.fromSource()
                .arity( 3 )
                .kernelName( "fallBackMatMul" )
                .kernelSource(
                        "_kernel void fallBackMatMul(                                                 " +
                                "   int widthA,                                                       " +
                                "   int heightA,                                                      " +
                                "   int widthB,                                                       " +
                                "   int heightB,                                                      " +
                                "   __global float* outputC, __global int *confC,                     " +
                                "   __global float* inputA,  __global int *confA,                     " +
                                "   __global float* inputB   __global int *confB,                     " +
                                ") {                                                                  " +
                                "   int prvConfC[32]; _cfg_of_cfg(prvConfC, prvConfC, rank);\n        " +
                                "   int prvConfA[32]; _cfg_of_cfg(prvConfA, prvConfA, rank);\n        " +
                                "   int prvConfB[32]; _cfg_of_cfg(prvConfB, prvConfB, rank);          " +
                                "   int row = get_global_id( 1 );                                     " +
                                "   int col = get_global_id(0);                                       " +
                                "   float sum = 0.0f;                                                 " +
                                "   for ( int i = 0; i < widthA; i++ ) {                              " +
                                "      sum += inputA[ row * widthA + i ] * inputB[ i * widthB + col ];" +
                                "   }                                                                 " +
                                "   outputC[ row * widthB * col ] = sum;                              " +
                                "}"
                )
                .build();


    }


    @Override
    public String stringify( String[] children ) {
        StringBuilder reconstructed = new StringBuilder();
        for ( int i = 0; i < children.length; ++i ) {
            reconstructed.append( children[ i ] );
            if ( i < children.length - 1 ) {
                reconstructed.append(" @ ");
            }
        }
        return "(" + reconstructed + ")";
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        return src[ 0 ].call( inputs, j );
    }
}
