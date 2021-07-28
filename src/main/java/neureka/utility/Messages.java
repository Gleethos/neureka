package neureka.utility;

import neureka.backend.api.Algorithm;
import neureka.backend.api.Operation;
import neureka.backend.api.OperationContext;
import neureka.devices.opencl.CLContext;
import org.slf4j.helpers.MessageFormatter;

/**
 *  A simple collection of prepared
 *  String messages which provide helpful messages for the logging
 *  backend and ultimately the user of this library.
 */
public class Messages
{
    public static final String ILLEGAL_OPERATION_STATE_ERROR =
                    "Unexpected '"+ Operation.class.getSimpleName()+"' state encountered:\n" +
                    "The operation '{}' String should not be null but was null!";

    public static final String OPERATION_LOADED_DEBUG =
                    "Operation: '{}' loaded!";

    public static final String CL_CONTEXT_NOT_CREATED_WARNING =
                    "OpenCL not available!\n" +
                    "Skipped creating and adding a new '"+ CLContext.class.getSimpleName()+"' " +
                    "to the current '"+ OperationContext.class.getSimpleName()+"'...";

    public static class Device {

        public static String couldNotFindSuitableAlgorithmFor( Class<?> type ) {
            return _format(
                        "No suitable '"+ Algorithm.class.getSimpleName()+"' found for device of type '{}'.",
                        type.getSimpleName()
                    );
        }

        public static String couldNotFindSuitableImplementationFor(
                Algorithm<?> algorithm,
                Class<?> type
        ) {
           return _format(
                   "No suitable implementation found for algorithm '{}' and device type '{}'.",
                   algorithm.getName(),
                   type.getSimpleName()
               );
        }

    }

    private static String _format( String withPlaceholders, Object... toBePutAtPlaceholders ) {
        return MessageFormatter.arrayFormat(withPlaceholders, toBePutAtPlaceholders).getMessage();
    }

}
