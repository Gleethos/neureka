package neureka.backend.api;

import neureka.internal.common.composition.AbstractComponentOwner;
import neureka.internal.common.composition.Component;

/**
 *  This is an internal class for managin gthe extension of any given {@link BackendContext} class.
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
