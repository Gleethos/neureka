package neureka.devices.opencl;

/**
 *  OpenCL related settings for the {@link CLContext} backend extension.
 */
public class CLSettings {

    private boolean _autoConvertToFloat = true;

    public boolean isAutoConvertToFloat() {
        return _autoConvertToFloat;
    }

    public CLSettings setAutoConvertToFloat(boolean autoConvertToFloat) {
        this._autoConvertToFloat = autoConvertToFloat;
        return this;
    }

}
