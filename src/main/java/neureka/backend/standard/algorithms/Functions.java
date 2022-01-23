package neureka.backend.standard.algorithms;

import neureka.backend.api.ImplementationFor;
import neureka.backend.api.implementations.AbstractImplementationFor;
import neureka.devices.host.CPU;

import java.util.ArrayList;
import java.util.List;

public class Functions<F extends Fun> {

    private final List<FunArray<F>> _functions = new ArrayList<>();

    public static <F extends Fun, P extends FunPair<F>> Builder<F> implementation(
            int arity, FunImplementation<F> composed
    ) {
        return new Builder<F>( arity, composed );
    }

    private Functions( List<FunArray<F>> fun ) {
        _functions.addAll(fun);
    }

    public <T extends F> FunArray<T> get( Class<T> type ) {
        for ( FunArray<F> p : _functions )
            if ( type.isAssignableFrom( p.getType() ) )
                return (FunArray<T>) p;

        throw new IllegalArgumentException("");
    }

    public static class Builder<F extends Fun>
    {
        private final List<FunArray<F>> _functions = new ArrayList<>();
        private final FunImplementation<F> _implementation;
        private final int _arity;

        private Builder(
                int arity, FunImplementation<F> composed
        ) {
            _arity = arity;
            _implementation = composed;
        }

        public <T extends F, P extends FunArray<T>> Builder<F> with( P fun ) {
            _functions.add((FunArray<F>) fun);
            return this;
        }

        public ImplementationFor<CPU> get() {
            return new AbstractImplementationFor<>(
                    call -> _implementation.get(call, new Functions<F>(_functions)),
                    _arity
            );
        }

    }

}
