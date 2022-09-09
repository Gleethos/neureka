package neureka.backend.api.annotations;

import neureka.backend.api.DeviceAlgorithm;
import neureka.backend.api.Operation;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;

@Retention(RetentionPolicy.RUNTIME)
public @interface Loadable {
    Class<? extends Operation> operation();
    Class<? extends DeviceAlgorithm<?>> algorithm();
}
