package neureka.backend.api;

import neureka.Component;
import neureka.ndim.AbstractComponentOwner;

public class Args extends AbstractComponentOwner<Args> {

    public <V, T extends Argument<V>> V findAndGet( Class<T> argumentClass ) {
        Argument<V> argument = find(argumentClass);
        if ( argument == null ) return null; else return argument.get();
    }

    @Override
    protected <T extends Component<Args>> T _setOrReject(T newComponent) {
        return newComponent;
    }

    @Override
    protected <T extends Component<Args>> T _removeOrReject(T newComponent) {
        return newComponent;
    }
}
