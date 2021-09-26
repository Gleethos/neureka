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

        RecursiveExecutor rja = (call, goDeeperWith) ->
        {
            Tsr<?>[] tsrs = call.getTensors();
            Device<?> device = call.getDevice();
            int d = call.getValOf( Arg.DerivIdx.class );
            Operation type = call.getOperation();

            Tsr<?> alternative = null;
            if ( tsrs.length > 3 ) {
                if ( d < 0 ) {
                    Tsr<?>[] reduction = new Tsr[]{tsrs[ 0 ], tsrs[ 1 ], tsrs[ 2 ]};
                    alternative = goDeeperWith.execute(
                                            ExecutionCall.of(reduction).andArgs(Arg.DerivIdx.of(d)).running(type).on(device)
                                    );
                    tsrs[ 0 ] = reduction[ 0 ];

                    reduction = Utility.offsetted(tsrs, 1);
                    alternative = goDeeperWith.execute(
                            ExecutionCall.of(reduction).andArgs(Arg.DerivIdx.of(d)).running(type).on(device)
                    );
                    tsrs[ 0 ] = reduction[ 0 ];
                }
                return alternative;
            }
            else
                return alternative;
        };

        FunAlgorithm simpleMatMulAlgorithm =
                        Algorithm.withName("simple_matmul")
                                    .setIsSuitableFor(
                                            call -> call.validate()
                                                        .all( t -> t.getNDConf() instanceof SimpleD2Configuration )
                                                        .estimation()
                                    )
                                    .setCanPerformBackwardADFor( call -> true )
                                    .setCanPerformForwardADFor(
                                        call -> {
                                            if ( call.getOperation().supports(Convolution.class) ) return false;
                                            if ( call.getOperation().getOperator().equals(",") ) return false; //Reshape
                                            Tsr<?> last = null;
                                            for ( Tsr<?> t : call.getTensors() ) {
                                                if ( last != null && !last.shape().equals(t.shape()) ) return false;
                                                last = t; // Note: shapes are cached!
                                            }
                                            return true;
                                        }
                                    )
                                    .setSupplyADAgentFor(
                                        ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                                        {
                                            if ( forward ) throw new IllegalArgumentException("Matrix multiplication of does not support forward-AD!");

                                            Function invX = new FunctionBuilder( Neureka.get().context() ).build( "I[ 0 ] @ I[ 1 ]", false );
                                            Tsr<?>[] inputs = call.getTensors();
                                            int d = (1 + call.getValOf( Arg.DerivIdx.class )) % 2;
                                            Tsr<?> deriv = inputs[ d ].T();
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
                                            if ( !caller.isFlat() ) return CalcUtil.defaultRecursiveExecution( caller, call );
                                            if ( call.getOperation().getOperator().equals("x") ) {

                                                Tsr<?>[] inputs = call.getTensors();
                                                Tsr<?>[] tsrs = new Tsr[]{null, inputs[ 0 ], inputs[ 1 ]};
                                                tsrs[ 0 ] = (call.getValOf( Arg.DerivIdx.class ) < 0)
                                                        ? Tsr.ofShape( Tsr.Utility.Indexing.shpOfCon(tsrs[ 1 ].getNDConf().shape(), tsrs[ 2 ].getNDConf().shape()) )
                                                        : null;

                                                for (Tsr<?> t : tsrs) if (t != null) t.setIsVirtual( false );
                                                CalcUtil.recursiveExecution(call.withTensors(tsrs), rja);
                                                return tsrs[ 0 ];
                                            } else {
                                                if (call.getValOf( Arg.DerivIdx.class ) < 0) {
                                                    Tsr<?>[] tensors = CalcUtil.srcActivation(call.getTensors(), call.getJ(), -1, 0, caller.getSubFunctions().toArray(new Function[0]));
                                                    Tsr.makeFit(tensors, caller.isDoingAD()); // This might not fit here... (fitting should probably be a setup thing...)
                                                    for ( Tsr<?> t : tensors ) t.setIsVirtual( false );
                                                    CalcUtil.recursiveExecution(
                                                                        ExecutionCall.of(tensors)
                                                                                        .andArgs(Arg.DerivIdx.of(0))
                                                                                        .running(call.getOperation())
                                                                                        .on(call.getDevice()),
                                                                        (executionCall, executor) -> null
                                                                );
                                                    if ( call.getOperation() == Neureka.get().context().getOperation("x>>") )
                                                        return tensors[ 2 ];
                                                    else
                                                        return tensors[ 0 ];
                                                }
                                            }
                                            return null;
                                        }
                                    )
                                    .setCallPreparation(
                                        call -> {
                                            Tsr<?>[] tsrs = call.getTensors();
                                            Device device = call.getDevice();
                                            if ( tsrs[ 0 ] == null ) // Creating a new tensor:
                                            {
                                                int[] shp = Tsr.Utility.Indexing.shpOfCon(tsrs[ 1 ].getNDConf().shape(), tsrs[ 2 ].getNDConf().shape());
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
                                    .build();

        setAlgorithm(
                simpleMatMulAlgorithm
                        .setImplementationFor(
                                HostCPU.class,
                                new HostImplementation(
                                        call ->
                                                call.getDevice().getExecutor()
                                                        .threaded (
                                                                call.getTsrOfType( Number.class, 0 ).size(),
                                                                ( start, end ) -> {} // TODO: Simple matmul without fancy indexing
                                                        ),
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
