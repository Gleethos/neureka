package neureka.backend.main.implementations.elementwise;

import neureka.Neureka;
import neureka.backend.main.implementations.ParsedCLImplementation;
import neureka.devices.opencl.KernelCode;

public class CLBiElementwise  extends ParsedCLImplementation
{
    public CLBiElementwise(  String postfix, String activationSource, String differentiationSource ) {
        super(
            call -> {
                int offset = (call.input( Number.class, 0 ) != null) ? 0 : 1;
                int gwz = (call.input( Number.class, 0 ) != null) ? call.input( Number.class, 0 ).size() : call.input( Number.class, 1 ).size();
                call.getDevice()
                        .getKernel(call)
                        .passAllOf( call.input( Number.class, offset ) )
                        .passAllOf( call.input( Number.class, offset + 1 ) )
                        .passAllOf( call.input( Number.class, offset + 2 ) )
                        .pass( call.input( Number.class, 0 ).rank() )
                        .pass( call.getDerivativeIndex() )
                        .call( gwz );

                return call.input( 0 );
            },
            -1,
            Neureka.get().utility().readResource("kernels/elementwise_template.cl"),
            activationSource,
            differentiationSource,
            postfix,
            kernelCode -> new KernelCode[]{kernelCode}
        );
    }
}
