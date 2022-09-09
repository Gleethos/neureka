package neureka.backend.api.annotations;

import neureka.devices.Device;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;

@Retention(RetentionPolicy.RUNTIME)
public @interface Backend {
    Class<? extends Device<?>> device();
}
