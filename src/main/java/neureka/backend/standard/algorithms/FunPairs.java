package neureka.backend.standard.algorithms;

import neureka.backend.api.Fun;
import neureka.backend.api.ImplementationFor;
import neureka.devices.host.CPU;

import java.util.ArrayList;
import java.util.List;

public class FunPairs<F extends Fun> {

    private final List<FunPair<F>> _functions = new ArrayList<>();

    public static <F extends Fun, P extends FunPair<F>> Builder<F> implementation(
            FunImplementation<F> composed
            //,
            //Function<CPU.RangeWorkload, ImplementationFor<CPU>> implementationProvider
    ) {
        return new Builder<F>( composed );
    }

    private FunPairs(List<FunPair<F>> fun) {
        _functions.addAll(fun);
    }

    public <T extends F> FunPair<T> get( Class<T> type ) {
        for ( FunPair<F> p : _functions )
            if ( type.isAssignableFrom(p.getType()) )
                return (FunPair<T>) p;

        throw new IllegalArgumentException("");
    }

    public static class Builder<F extends Fun>
    {
        private final List<FunPair<F>> _functions = new ArrayList<>();
        FunImplementation<F> _implementation;

        private Builder(
                FunImplementation<F> composed
        ) {
            _implementation = composed;
        }

        public <P extends FunPair<F>> Builder<F> with(Class<F> type, P fun) {
            _functions.add(fun);
            return this;
        }

        public <T extends F, P extends FunPair<T>> Builder<F> with(P fun) {
            _functions.add((FunPair<F>) fun);
            return this;
        }

        public ImplementationFor<CPU> get() {
            return call -> _implementation.get(call, new FunPairs<F>(_functions));
        }

    }

}
