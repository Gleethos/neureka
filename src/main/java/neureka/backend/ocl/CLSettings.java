package neureka.backend.ocl;

/**
 *  OpenCL related settings for the {@link CLBackend} backend extension.
 */
public class CLSettings {

    private boolean _autoConvertToFloat = false;

    public boolean isAutoConvertToFloat() { return _autoConvertToFloat; }

    public CLSettings setAutoConvertToFloat(boolean autoConvertToFloat) {
        _autoConvertToFloat = autoConvertToFloat;
        return this;
    }

    public void reset() {
        _autoConvertToFloat = false;
    }

}
