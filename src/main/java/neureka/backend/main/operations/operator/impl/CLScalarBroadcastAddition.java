package neureka.backend.main.operations.operator.impl;

import neureka.Tsr;
import neureka.backend.main.algorithms.Scalarization;
import neureka.backend.main.implementations.CLImplementation;
import neureka.backend.main.implementations.ParsedCLImplementation;
import neureka.calculus.args.Arg;

public class CLScalarBroadcastAddition extends ParsedCLImplementation
{
    public CLScalarBroadcastAddition(String id) {
        super(
            CLImplementation
            .compiler()
            .arity( 3 )
            .kernelSource( Scalarization.getKernelSource() )
            .activationSource( "output = input1 + value;\n" )
            .differentiationSource( "output = 1;\n" )
            .kernelPostfix( id )
            .execution(
                call -> {
                    assert call.arity() == 3;
                    if ( call.getDerivativeIndex() == 0 )
                         return Tsr.of( call.input(1).shape(), 1d ).getUnsafe().setIsIntermediate( true );
                    else if ( call.getDerivativeIndex() == 1 )
                        return Tsr.of( call.input( 2 ).shape(), 1d ).getUnsafe().setIsIntermediate( true );
                    else {
                        int gwz = call.input(Number.class, 0).size();
                        float value = call.input(Number.class, 2).item(0).floatValue();
                        call.getDevice()
                            .getKernel(call)
                            .passAllOf(call.input(Number.class, 0))
                            .passAllOf(call.input(Number.class, 1))
                            .pass(value)
                            .pass(call.input(Number.class, 0).rank())
                            .pass(call.getValOf(Arg.DerivIdx.class))
                            .call(gwz);
                    }
                    return call.input( 0 );
                }
            )
        );
    }
}
