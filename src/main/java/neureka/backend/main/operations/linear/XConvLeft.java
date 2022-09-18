package neureka.backend.main.operations.linear;

import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.NDConvolution;
import neureka.backend.main.implementations.convolution.CLConvolution;
import neureka.backend.main.implementations.convolution.CPUConvolution;
import neureka.backend.main.operations.ConvUtil;
import neureka.calculus.Function;
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
        setAlgorithm( NDConvolution.class,
            ConvUtil.createDeconvolutionFor(((char) 171) + "x")
            .setImplementationFor(
                CPU.class,
                new CPUConvolution()
            )
            .setImplementationFor(
                OpenCLDevice.class,
                new CLConvolution( this.getIdentifier() )
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
