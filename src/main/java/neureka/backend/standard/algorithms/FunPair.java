package neureka.backend.standard.algorithms;

public class FunPair<T> {

    private final Class<T> _type;
    private final T _a;
    private final T _d;

    public FunPair(T a, T d) {
        _a = a;
        _d = d;
        _type = (Class<T>) _a.getClass();
    }

    public T get(int d) { return ( d < 0 ? _a : _d ); }

    public Class<T> getType() {
        return _type;
    }

}
