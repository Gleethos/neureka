package neureka.backend.main.implementations.broadcast;

import neureka.backend.main.algorithms.Scalarization;
import neureka.backend.main.implementations.CLImplementation;
import neureka.backend.main.implementations.ParsedCLImplementation;
import neureka.calculus.args.Arg;

public class CLScalarBroadcastDivision extends ParsedCLImplementation
{
    public CLScalarBroadcastDivision(String id) {
        super(
            CLImplementation
            .compiler()
            .arity( 3 )
            .kernelSource( Scalarization.getKernelSource() )
            .activationSource( "output = input1 / value;\n" )
            .differentiationSource(
                "if (d==0) {                                       \n" +
                "    output = 1/value;                             \n" +
                "} else {                                          \n" +
                "    output = -value /(float)pow(input1, 2.0f);    \n" +
                "}                                                 \n"
            )
            .kernelPostfix( id )
            .execution(
                call -> {
                    int offset = (call.input( Number.class, 2 ).isVirtual() || call.input( Number.class, 2 ).size() == 1)?1:0;
                    int gwz = call.input( Number.class, 0 ).size();
                    call.getDevice().getKernel(call)
                            .passAllOf(call.input( Number.class, 0 ))
                            .passAllOf(call.input( Number.class, 0 ))
                            .pass( call.input( Number.class, 1 + offset ).at( 0 ).get().floatValue() )
                            .pass( call.input( Number.class, 0 ).rank() )
                            .pass( call.getValOf( Arg.DerivIdx.class ) )
                            .call( gwz );

                    return call.input( 0 );
                }
            )
        );
    }
}
