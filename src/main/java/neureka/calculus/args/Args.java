package neureka.calculus.args;

import neureka.common.composition.Component;
import neureka.common.composition.AbstractComponentOwner;

public class Args extends AbstractComponentOwner<Args> {

    public static Args of( Arg<?>... arguments ) {
        Args args = new Args();
        for ( Arg<?> arg : arguments ) {
            if ( arg != null ) args.set(arg);
        }
        return args;
    }

    public <V, T extends Arg<V>> V valOf( Class<T> argumentClass ) {
        Arg<V> argument = get(argumentClass);
        if ( argument == null ) return null; else return argument.get();
    }

    public <V, T extends Arg<V>> V valOfOr( Class<T> argumentClass, V fallback ) {
        Arg<V> argument = get(argumentClass);
        if ( argument == null ) return fallback; else return argument.get();
    }

    @Override
    protected <T extends Component<Args>> T _setOrReject( T newComponent ) {
        return newComponent;
    }

    @Override
    protected <T extends Component<Args>> T _removeOrReject(T newComponent) {
        return newComponent;
    }
}
