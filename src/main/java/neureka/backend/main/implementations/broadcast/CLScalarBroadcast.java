package neureka.backend.main.implementations.broadcast;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.main.implementations.ParsedCLImplementation;
import neureka.calculus.args.Arg;

public class CLScalarBroadcast extends ParsedCLImplementation
{
    public CLScalarBroadcast(
        String postfix, String activation, String derivation
    ) {
        super(
            call->{
                Tsr<Number> t = call.input( Number.class, 0 );
                int gwz = t.size();
                call.getDevice()
                        .getKernel(call)
                        .passAllOf( t )
                        .passAllOf( t )
                        .pass( call.input( Number.class, 1 ).at(0).get().floatValue() )
                        .pass( t.rank() )
                        .pass( call.getValOf( Arg.DerivIdx.class ) )
                        .call( gwz );

                return call.input(0);
            },
            2,
            Neureka.get().utility().readResource("kernels/scalarization_template.cl"),
            activation,
            derivation,
            postfix
        );
    }
}
