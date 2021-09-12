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

    public static class OpenCL {

        public static String clContextCreationFailed() {
            return _format(
                    "OpenCL not available!\n" +
                            "Skipped creating and adding a new '"+ CLContext.class.getSimpleName()+"' " +
                            "to the current '"+ OperationContext.class.getSimpleName()+"'..."
            );
        }


        public enum Tips {
            UBUNTU(
                    "Try executing the following command to install OpenCL: 'sudo apt install ocl-icd-opencl-dev'.\n",
                    "In order to allow OpenCL to find your GPUs consider executing 'sudo ubuntu-drivers autoinstall'!\n"
            ),
            FEDORA(
                    "Try executing the following command to install OpenCL: 'sudo dnf install ocl-icd-devel'.\n",
                    "In order to allow OpenCL to find your GPUs consider installing or updating your device drivers!\n"
            ),
            WINDOWS(
                    "Try to install OpenCL and the latest drivers of your GPU (Or other SIMD devices).",
                    ""
            ),
            UNKNOWN("", "");

            public final String HOW_TO_INSTALL_OPENCL;
            private final String HOW_TO_INSTALL_OPENCL_DRIVERS;

            Tips( String howToInstallOpenCL, String howToInstallDrivers ) {
                HOW_TO_INSTALL_OPENCL = howToInstallOpenCL;
                HOW_TO_INSTALL_OPENCL_DRIVERS = howToInstallDrivers;
            }

            public String bootstrapTip() {
                return !HOW_TO_INSTALL_OPENCL.isEmpty()
                                ? ("[Tip]: "+ HOW_TO_INSTALL_OPENCL +""+ HOW_TO_INSTALL_OPENCL_DRIVERS)
                                : ("");
            }
        }

    }

    public static class Operations {

        public static String illegalStateFor( String type ) {
            return _format(
                    "Unexpected '"+ Operation.class.getSimpleName()+"' state encountered:\n" +
                    "The operation '{}' String should not be null but was null!",
                    type
            );
        }

        public static String loaded( Operation operation ) {
            return _format(
                    "Operation: '{}' loaded!",
                    operation.getFunction()
            );
        }

    }

    public static class Devices {

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
