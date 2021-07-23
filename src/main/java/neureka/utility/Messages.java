package neureka.utility;

import neureka.backend.api.Operation;
import neureka.backend.api.OperationContext;
import neureka.devices.opencl.CLContext;

/**
 *  A simple collection of public static final and pre-instantiated
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

}
