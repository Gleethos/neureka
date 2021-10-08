package neureka.backend.standard.operations.linear;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.Algorithm;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.FunAlgorithm;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.backend.standard.implementations.HostImplementation;
import neureka.calculus.CalcUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import neureka.devices.host.HostCPU;
import neureka.devices.opencl.OpenCLDevice;
import neureka.ndim.config.types.simple.SimpleD2Configuration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MatMul extends AbstractOperation
{
    private static Logger _LOG = LoggerFactory.getLogger(MatMul.class);

    public MatMul()
    {
        super(
                new OperationBuilder()
                        .setFunction(         "matMul"    )
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
                                                        .allNotNull( t -> Number.class.isAssignableFrom(t.getValueClass()) )
                                                        .getEstimator()
                                                            .goodIfAnyNonNull( t -> t.getNDConf() instanceof SimpleD2Configuration )
                                                            .badIfAnyNonNull( t -> !( t.getNDConf() instanceof SimpleD2Configuration ) )
                                                            .getEstimation()
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
                                            Tsr<?> deriv = inputs[ d ].T().clone(); // We need to clone it to make it have a simple nd configuration...
                                            deriv.to(call.getDevice());
                                            return ADAgent.of( deriv )
                                                            .setBackward( (node, error) -> {
                                                                if ( d == 1 )
                                                                    return invX.execute( error, deriv );
                                                                else
                                                                    return invX.execute( deriv, error );
                                                            });
                                        }
                                    )
                                    .setExecutionDispatcher(
                                        ( caller, call ) -> {
                                            if ( !caller.isFlat() )
                                                return CalcUtil.defaultRecursiveExecution( caller, call );

                                            Tsr<?>[] tensors = CalcUtil.srcActivation(call.getTensors(), call.getJ(), -1, 1, caller.getSubFunctions().toArray(new Function[0]));
                                            for ( Tsr<?> t : tensors ) if ( t != null ) t.setIsVirtual( false );
                                            ExecutionCall<Device<Object>> preparedCall = (ExecutionCall<Device<Object>>) call.getAlgorithm().prepare( call.withTensors(tensors) );
                                            return call.getAlgorithm()
                                                        .getImplementationFor((Class<Device<Object>>) call.getDevice().getClass())
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
                                            _autoClone( tsrs );
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
                                        .kernelName( "simple_matMul" )
                                        .kernelSource(
                                                "__kernel void simple_matMul(                                          \n" +
                                                "       const int M, const int N, const int K,                        \n" +
                                                "       const __global float* A,                                      \n" +
                                                "       const __global float* B,                                      \n" +
                                                "       __global float* C                                             \n" +
                                                ") {                                                                  \n" +
                                                "    const int globalRow = get_global_id(0); // Row ID of C (0..M)    \n" +
                                                "    const int globalCol = get_global_id(1); // Col ID of C (0..N)    \n" +
                                                "                                                                     \n" +
                                                "    // Compute a single element (loop over K)                        \n" +
                                                "    float acc = 0.0f;                                                \n" +
                                                "    for ( int k = 0; k < K; k++ ) {                                  \n" +
                                                "        acc += A[k + globalRow*K] * B[globalCol + k*N];              \n" +
                                                "    }                                                                \n" +
                                                "    // Store the result                                              \n" +
                                                "    C[globalCol + globalRow*N] = acc;                                \n" +
                                                "}                                                                    \n"
                                        )
                                        .lambda( call -> {
                                            int M = call.getTensors()[1].shape(0);
                                            int N = call.getTensors()[2].shape(1);
                                            int K = call.getTensors()[1].shape(1);
                                            call.getDevice()
                                                    .getKernel(call)
                                                    .pass(M) // M
                                                    .pass(N) // N
                                                    .pass(K) // K
                                                    .pass(call.getTsrOfType(Number.class, 1))
                                                    .pass(call.getTsrOfType(Number.class, 2))
                                                    .pass(call.getTsrOfType(Number.class, 0))
                                                    .call(new long[]{M,N}, null);
                                        } )
                                        .build()
                        )
        );
        /*
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
        */

    }

    private static void _autoClone( Tsr<?>[] tensors ) {
        for ( int i = 0; i < tensors.length; i++ ) {
            if ( !(tensors[i].getNDConf() instanceof SimpleD2Configuration) ) {
                _LOG.warn("Auto cloning a tensor which does not have a simple ND configuration...");
                tensors[i] = tensors[i].clone();
            }
        }
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
