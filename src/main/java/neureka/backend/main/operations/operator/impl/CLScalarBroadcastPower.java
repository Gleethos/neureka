package neureka.backend.main.operations.operator.impl;

import neureka.backend.main.algorithms.Scalarization;
import neureka.backend.main.implementations.CLImplementation;
import neureka.backend.main.implementations.ParsedCLImplementation;
import neureka.calculus.args.Arg;

public class CLScalarBroadcastPower extends ParsedCLImplementation
{
    public CLScalarBroadcastPower(String id) {
        super(
            CLImplementation
                .compiler()
                .arity( 3 )
                .kernelSource( Scalarization.getKernelSource() )
                .activationSource( "output = pow( input1, value );" )
                .differentiationSource(
                    "   if ( d == 0 )                                            \n" +
                    "       output = value * pow( input1, value - (float) 1 );   \n" +
                    "   else                                                     \n" +
                    "       output = pow( input1, value ) * log( value );        \n"
                )
                .kernelPostfix( id )
                .execution(
                    call -> {
                        int offset = (call.input( Number.class, 2 ).isVirtual() || call.input( Number.class, 2 ).size() == 1)?1:0;
                        int gwz = call.input( Number.class, 0 ).size();
                        call.getDevice()
                            .getKernel( call )
                            .passAllOf(call.input( Number.class, 0 ))
                            .passAllOf(call.input( Number.class, 0 ))
                            .pass( call.input( Number.class, 1 + offset ).at( 0 ).get().floatValue() )
                            .pass( call.input( Number.class, 0 ).rank() )
                            .pass( call.getValOf( Arg.DerivIdx.class ) )
                            .call( gwz );

                        return call.input(0);
                    }
                )
        );
    }
}
