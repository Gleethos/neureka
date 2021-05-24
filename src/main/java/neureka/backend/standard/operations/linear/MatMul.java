package neureka.backend.standard.operations.linear;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.DefaultADAgent;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Algorithm;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.Operation;
import neureka.backend.api.operations.OperationContext;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Convolution;
import neureka.backend.standard.algorithms.GenericAlgorithm;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.backend.standard.implementations.HostImplementation;
import neureka.calculus.Function;
import neureka.calculus.assembly.FunctionBuilder;
import neureka.devices.Device;
import neureka.devices.host.HostCPU;
import neureka.devices.opencl.OpenCLDevice;

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

        Algorithm.RecursiveJunctor rja = (call, goDeeperWith)->
        {
            Tsr[] tsrs = call.getTensors();
            Device device = call.getDevice();
            int d = call.getDerivativeIndex();
            Operation type = call.getOperation();

            Tsr alternative = null;
            if (tsrs.length > 3) {
                if ( d < 0 ) {
                    Tsr[] reduction = new Tsr[]{tsrs[ 0 ], tsrs[ 1 ], tsrs[ 2 ]};
                    alternative = goDeeperWith.apply(
                            ExecutionCall.builder()
                                    .device(device)
                                    .tensors(reduction)
                                    .derivativeIndex(d)
                                    .operation(type)
                                    .build()
                    );
                    tsrs[ 0 ] = reduction[ 0 ];

                    reduction = Utility.offsetted(tsrs, 1);
                    alternative = goDeeperWith.apply(
                            ExecutionCall.builder()
                                    .device(device)
                                    .tensors(reduction)
                                    .derivativeIndex(d)
                                    .operation(type)
                                    .build()
                    );
                    tsrs[ 0 ] = reduction[ 0 ];
                }
                return alternative;
            } else
                return alternative;

        };

        DefaultOperatorCreator<TertiaryNDIConsumer> convolutionNDICreator =
                ( inputs, d ) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    double[] t2_val = inputs[ 2 ].value64();
                    if ( d < 0 ) {
                        return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ] * t2_val[t2Idx.i()];
                    } else {
                        return ( t0Idx, t1Idx, t2Idx ) -> {
                            if (d == 0) return t2_val[t2Idx.i()];
                            else return t1_val[ t1Idx.i() ];
                        };
                    }
                };

        DefaultOperatorCreator<TertiaryNDAConsumer> convolutionCreator =
                ( inputs, d ) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    double[] t2_val = inputs[ 2 ].value64();
                    if ( d < 0 ) {
                        return ( t0Idx, t1Idx, t2Idx ) -> t1_val[inputs[ 1 ].indexOfIndices( t1Idx )] * t2_val[inputs[ 2 ].indexOfIndices(t2Idx)];
                    } else {
                        return ( t0Idx, t1Idx, t2Idx ) -> {
                            if (d == 0) return t2_val[inputs[ 2 ].indexOfIndices(t2Idx)];
                            else return t1_val[inputs[ 1 ].indexOfIndices( t1Idx )];
                        };
                    }
                };


        GenericAlgorithm convolution = new GenericAlgorithm("matmul")
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
                            //Tsr ctxDerivative = (Tsr) call.getAt("derivative");
                            if ( forward ) throw new IllegalArgumentException("Matrix multiplication of does not support forward-AD!");

                            Function invX = new FunctionBuilder(OperationContext.get()).build( "I[ 0 ] @ I[ 1 ]", false );
                            Tsr[] inputs = call.getTensors();
                            int d = call.getDerivativeIndex();
                            Tsr deriv = inputs[1+d].T();//f.derive( inputs, d );
                            return new DefaultADAgent( deriv )
                                    .setForward( (node, forwardDerivative ) -> null )
                                    .setBackward( (t, error) -> invX.call(new Tsr[]{ error, deriv }) );
                        }
                )
                .setHandleInsteadOfDevice(
                        ( caller, call ) -> {
                            if ( !caller.isFlat() ) return null;
                            if ( call.getOperation().getOperator().equals("x") ) {

                                Tsr[] inputs = call.getTensors();
                                Tsr[] tsrs = new Tsr[]{null, inputs[ 0 ], inputs[ 1 ]};
                                tsrs[ 0 ] = (call.getDerivativeIndex() < 0)
                                        ? new Tsr( Tsr.Utility.Indexing.shpOfCon(tsrs[ 1 ].getNDConf().shape(), tsrs[ 2 ].getNDConf().shape()) )
                                        : null;

                                for (Tsr t : tsrs) if (t != null) t.setIsVirtual( false );
                                call.getDevice().execute(call.withTensors(tsrs));
                                return tsrs[ 0 ];
                            } else {
                                if (call.getDerivativeIndex() < 0) {
                                    Tsr[] tsrs = caller.srcActivation(call.getTensors(), call.getJ(), -1, 0);
                                    Tsr.makeFit(tsrs, caller.isDoingAD()); // This might not fit here... (fitting should probably be a setup thing...)
                                    for ( Tsr t : tsrs ) t.setIsVirtual( false );
                                    call.getDevice().execute(
                                            ExecutionCall.builder()
                                                    .device(call.getDevice())
                                                    .tensors( tsrs )
                                                    .derivativeIndex( 0 )
                                                    .operation( call.getOperation() )
                                                    .build()
                                    );
                                    if ( call.getOperation().getId() == OperationContext.get().instance("x>>").getId()) return tsrs[ 2 ];
                                    else return tsrs[ 0 ];
                                }
                            }
                            return null;
                        }
                )
                .setHandleRecursivelyAccordingToArity( rja )
                .setInstantiateNewTensorsForExecutionIn(
                        call -> {
                            Tsr[] tsrs = call.getTensors();
                            Device device = call.getDevice();
                            if ( tsrs[ 0 ] == null ) // Creating a new tensor:
                            {
                                int[] shp = Tsr.Utility.Indexing.shpOfCon(tsrs[ 1 ].getNDConf().shape(), tsrs[ 2 ].getNDConf().shape());
                                Tsr output = new Tsr( shp, 0.0 );
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
                GenericAlgorithm.class,
                convolution
                        .setImplementationFor(
                                HostCPU.class,
                                new HostImplementation(
                                        call ->
                                                call.getDevice().getExecutor()
                                                        .threaded (
                                                                call.getTsrOfType( Number.class, 0 ).size(),
                                                                (Neureka.instance().settings().indexing().isUsingArrayBasedIndexing())
                                                                        ? ( start, end ) ->
                                                                        Convolution.convolve (
                                                                                call.getTsrOfType( Number.class, 0 ), call.getTsrOfType( Number.class, 1 ), call.getTsrOfType( Number.class, 2 ),
                                                                                call.getDerivativeIndex(), start, end,
                                                                                convolutionCreator.create(
                                                                                        call.getTensors(),
                                                                                        -1//call.getDerivativeIndex()
                                                                                )
                                                                        )
                                                                        :  ( start, end ) ->
                                                                        Convolution.convolve (
                                                                                call.getTsrOfType( Number.class, 0 ), call.getTsrOfType( Number.class, 1 ), call.getTsrOfType( Number.class, 2 ),
                                                                                call.getDerivativeIndex(), start, end,
                                                                                convolutionNDICreator.create(
                                                                                        call.getTensors(),
                                                                                        -1//call.getDerivativeIndex()
                                                                                )
                                                                        )
                                                        ),
                                        3
                                )
                        )
                        .setImplementationFor(
                                OpenCLDevice.class,
                                CLImplementation.fromSource()
                                        .arity( 3 )
                                        .kernelName( "" )
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
    public String asDerivative( Function[] children, int d ) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        return src[ 0 ].call( inputs, j );
    }
}
