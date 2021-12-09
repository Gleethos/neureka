package neureka.backend.api;

import neureka.common.composition.AbstractComponentOwner;
import neureka.common.composition.Component;

/**
 *  This is an internal class for managing the extension of any given {@link BackendContext} class.
 */
public class Extensions extends AbstractComponentOwner<Extensions> {

    @Override
    protected <T extends Component<Extensions>> T _setOrReject(T newComponent) {
        return newComponent;
    }

    @Override
    protected <T extends Component<Extensions>> T _removeOrReject(T newComponent) {
        return newComponent;
    }
}
