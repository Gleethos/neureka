package neureka.backend.api;

import neureka.common.composition.Component;

public interface BackendExtension extends Component<Extensions> {

    /**
     *  Tells this extension to dispose itself.
     *  One should not use a {@link BackendExtension} after it was disposed!
     */
    void dispose();

}
