package neureka.backend.main.operations.operator.impl;

import neureka.Neureka;
import neureka.backend.main.implementations.CLImplementation;
import neureka.backend.main.implementations.ParsedCLImplementation;
import neureka.calculus.args.Arg;

public class CLBroadcast extends ParsedCLImplementation
{
    protected CLBroadcast(String postfix, String forward, String backward) {
        super(
            CLImplementation.compiler()
            .arity( 3 )
            .kernelSource( Neureka.get().utility().readResource("kernels/broadcast_template.cl") )
            .activationSource( forward )
            .differentiationSource( backward )
            .kernelPostfix( postfix )
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
        );
    }
}
