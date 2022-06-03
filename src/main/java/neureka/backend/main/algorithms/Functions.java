package neureka.backend.main.algorithms;

import neureka.backend.api.Call;
import neureka.backend.api.ImplementationFor;
import neureka.backend.api.template.implementations.AbstractImplementationFor;
import neureka.backend.main.algorithms.internal.Fun;
import neureka.backend.main.algorithms.internal.FunImplementation;
import neureka.backend.main.algorithms.internal.FunTuple;
import neureka.backend.main.algorithms.internal.FunWorkloadFinder;
import neureka.devices.host.CPU;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 *  A simple container for various tuples of scalar based functions for specific data types,
 *  which will be used by various algorithms like {@link Activation} and
 *  {@link Broadcast} to apply them to many elements in parallel.
 *  Instances of this will be constructed through the
 *  {@link Functions.Builder} which exposes the {@link Functions.Builder#with(FunTuple)} method
 *  for registering {@link FunTuple}s.
 *
 * @param <F>
 */
public final class Functions<F extends Fun> {

    private final List<FunTuple<F>> _functions = new ArrayList<>();


    public static <F extends Fun> Builder<F> implementation(
            int arity, FunWorkloadFinder<F> composed
    ) {
        return new Builder<F>(
                arity,
                call -> call.input( 0 ).size(),
                composed
        );
    }

    public static <F extends Fun> Builder<F> implementation(
            int arity, Function<Call<?>, Integer> workSizeSupplier, FunWorkloadFinder<F> composed
    ) {
        return new Builder<F>( arity, workSizeSupplier, composed );
    }


    private Functions( List<FunTuple<F>> fun ) { _functions.addAll(fun); }


    public <T extends F> FunTuple<T> get(Class<T> type ) {
        for ( FunTuple<F> p : _functions )
            if ( type.isAssignableFrom( p.getType() ) )
                return (FunTuple<T>) p;

        throw new IllegalArgumentException("Function of type '"+type.getSimpleName()+"' not found!");
    }

    public static class Builder<F extends Fun>
    {
        private final List<FunTuple<F>> _functions = new ArrayList<>();
        private final FunImplementation<F> _implementation;
        private final int _arity;

        private Builder(
                int arity,
                Function<Call<?>, Integer> workSizeSupplier,
                FunWorkloadFinder<F> composed
        ) {
            _arity = arity;
            _implementation =
                    (call, pairs) ->
                            call.getDevice()
                                    .getExecutor()
                                    .threaded(
                                            workSizeSupplier.apply(call),
                                            composed.get( call, pairs )
                                    );
        }

        public <T extends F, P extends FunTuple<T>> Builder<F> with( P fun ) {
            _functions.add((FunTuple<F>) fun);
            return this;
        }

        public ImplementationFor<CPU> get() {
            return new AbstractImplementationFor<>(
                    call -> _implementation.get( call, new Functions<F>( _functions ) ),
                    _arity
                );
        }

    }

}
