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
import neureka.backend.standard.implementations.CLImplementation;
import neureka.backend.standard.implementations.HostImplementation;
import neureka.backend.standard.operations.ConvUtil;
import neureka.backend.standard.operations.JunctionUtil;
import neureka.calculus.Function;
import neureka.calculus.assembly.FunctionBuilder;
import neureka.devices.Device;
import neureka.devices.host.HostCPU;
import neureka.devices.opencl.OpenCLDevice;

public class XConv extends AbstractOperation
{

    public XConv()
    {
        super(
                new OperationBuilder()
                        .setFunction(         "multiply"    )
                        .setOperator(         "x"    )
                        .setArity(            2          )
                        .setIsOperator(       true       )
                        .setIsIndexer(        false       )
                        .setIsDifferentiable( true        )
                        .setIsInline(         false       )
        );

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

        Convolution convolution = ConvUtil.createDeconvolutionFor(this);

        setAlgorithm(
                Convolution.class,
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
                            CLImplementation.compiler()
                                    .arity( 3 )
                                    .kernelSource( convolution.getKernelSource() )
                                    .activationSource( "value = src1 * src2;\n" )
                                    .differentiationSource( "value += handle * drain;\n" )
                                    .kernelPostfix( this.getFunction() )
                                    .execution(
                                            call -> {
                                                int offset = ( call.getTsrOfType( Number.class, 0 ) != null ) ? 0 : 1;
                                                int gwz = ( call.getTsrOfType( Number.class, 0 ) != null ) ? call.getTsrOfType( Number.class, 0 ).size() : call.getTsrOfType( Number.class, 1 ).size();
                                                call.getDevice().getKernel(call)
                                                        .pass( call.getTsrOfType( Number.class, offset ) )
                                                        .pass( call.getTsrOfType( Number.class, offset + 1 ) )
                                                        .pass( call.getTsrOfType( Number.class, offset + 2 ) )
                                                        .pass( call.getTsrOfType( Number.class, 0 ).rank() )
                                                        .pass( call.getDerivativeIndex() ) //call.getDerivativeIndex()
                                                        .call( gwz );
                                            }
                                    )
                                    .build()
                        )
        );

        new AbstractOperation(
                new OperationBuilder()
                        .setFunction(         "inv_convolve_mul_left"    )
                        .setOperator(         ((char) 171) + "x"         )
                        .setArity(            3                         )
                        .setIsOperator(       true        )
                        .setIsIndexer(        false       )
                        .setIsDifferentiable( false       )
                        .setIsInline(         false       )
        ) {
            @Override
            public String stringify(String[] children) {
                StringBuilder reconstructed = new StringBuilder();
                for ( int i = 0; i < children.length; ++i ) {
                    reconstructed.append( children[ i ] );
                    if ( i < children.length - 1 ) {
                        reconstructed.append(" "+((char) 171) + "x ");
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
        .setAlgorithm(Convolution.class, convolution);

        new AbstractOperation(
                new OperationBuilder()
                        .setFunction(         "inv_convolve_mul_right"    )
                        .setOperator(         "x" + ((char) 187)         )
                        .setArity(            3                         )
                        .setIsOperator(       true        )
                        .setIsIndexer(        false       )
                        .setIsDifferentiable( false       )
                        .setIsInline(         false       )
                ) {
            @Override
            public String stringify(String[] children) {
                StringBuilder reconstructed = new StringBuilder();
                for ( int i = 0; i < children.length; ++i ) {
                    reconstructed.append( children[ i ] );
                    if ( i < children.length - 1 ) {
                        reconstructed.append(" x" + ((char) 187)+" ");
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
                return 0;
            }
        }
        .setAlgorithm( Convolution.class, convolution );




    }


    @Override
    public String stringify( String[] children ) {
        StringBuilder reconstructed = new StringBuilder();
        for ( int i = 0; i < children.length; ++i ) {
            reconstructed.append( children[ i ] );
            if ( i < children.length - 1 ) {
                reconstructed.append(" x ");
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
