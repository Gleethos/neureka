package neureka.backend.main.algorithms.internal;

final class FunPair<T extends Fun> implements FunTuple<T> {

    private final Class<T> _type;
    private final T _a;
    private final T _d;

    public FunPair( T a, T d ) {
        _a = a;
        _d = d;
        _type = (Class<T>) _a.getClass();
    }

    @Override
    public T get( int derivativeIndex ) { return ( derivativeIndex < 0 ? _a : _d ); }

    @Override
    public Class<T> getType() { return _type; }

}
