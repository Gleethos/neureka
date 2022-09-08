package neureka.backend.main.operations.linear;

import neureka.Neureka;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.Convolution;
import neureka.backend.main.implementations.CLImplementation;
import neureka.backend.main.operations.ConvUtil;
import neureka.backend.main.implementations.convolution.CPUXConv;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;

public class XConvLeft extends AbstractOperation {

    public XConvLeft() {
        super(
                new OperationBuilder()
                        .identifier(         "inv_convolve_mul_left"    )
                        .operator(         ((char) 171) + "x"         )
                        .arity(            3                         )
                        .isOperator(       true        )
                        .isIndexer(        false       )
                        .isDifferentiable( false       )
                        .isInline(         false       )
        );
        setAlgorithm( Convolution.class,
            ConvUtil.createDeconvolutionFor(((char) 171) + "x")
            .setImplementationFor(
                CPU.class,
                new CPUXConv()
            )
            .setImplementationFor(
                OpenCLDevice.class,
                CLImplementation.compiler()
                    .arity( 3 )
                    .kernelSource( Neureka.get().utility().readResource("kernels/convolution_template.cl") )
                    .activationSource( "value = src1 * src2;\n" )
                    .differentiationSource( "value += handle * drain;\n" )
                    .kernelPostfix( this.getIdentifier() )
                    .execution(
                        call -> {
                            int offset = ( call.input( Number.class, 0 ) != null ) ? 0 : 1;
                            int gwz = ( call.input( Number.class, 0 ) != null ) ? call.input( Number.class, 0 ).size() : call.input( Number.class, 1 ).size();
                            call.getDevice()
                                .getKernel(call)
                                .passAllOf( call.input( Number.class, offset ) )
                                .passAllOf( call.input( Number.class, offset + 1 ) )
                                .passAllOf( call.input( Number.class, offset + 2 ) )
                                .pass( call.input( Number.class, 0 ).rank() )
                                .pass( call.getValOf( Arg.DerivIdx.class ) )
                                .call( gwz );

                            return call.input( 0 );
                        }
                    )
                    .build()
            )
        );
    }

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
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        return src[ 0 ].call( inputs, j );
    }

}
