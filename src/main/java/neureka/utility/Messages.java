package neureka.utility;

import neureka.backend.api.Algorithm;
import neureka.backend.api.Operation;
import neureka.backend.api.OperationContext;
import neureka.devices.opencl.CLContext;
import org.slf4j.helpers.MessageFormatter;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.StringReader;
import java.util.Properties;

/**
 *  A simple collection of prepared
 *  String messages which provide helpful messages for the logging
 *  backend and ultimately the user of this library.
 */
public class Messages
{
    /**
     * OpenCL specific messages and tips.
     */
    public static class OpenCL {

        public static String clContextCreationFailed() {
            return _format(
                    "OpenCL not available!\n" +
                            "Skipped creating and adding a new '"+ CLContext.class.getSimpleName()+"' " +
                            "to the current '"+ OperationContext.class.getSimpleName()+"'...\n" +
                            findTip().bootstrapTip()
            );
        }

        public static String clContextCouldNotFindAnyDevices() {
            return _format(
                    "OpenCL could not detect any devices in the current '"+ CLContext.class.getSimpleName()+"'.\n" +
                            findTip().HOW_TO_INSTALL_OPENCL_DRIVERS
            );
        }

        public static Messages.OpenCL.Tips findTip() {
            /*
                // Check lib: $ ls -l /usr/lib/libOpenCL*

                       UBUNTU:
                       $ sudo apt update
                       $ sudo apt install ocl-icd-opencl-dev
                       // Now libOpenCL.so should be located at /usr/lib/x86_64-linux-gnu/libOpenCL.so
                       $ sudo ubuntu-drivers autoinstall
                       ///////////////////
                       Switch ing from nvidia to amd:
                       The following command will remove the proprietary Nvidia driver:

                           $ sudo dpkg -P $(dpkg -l | grep nvidia-driver | awk '{print $2}')
                           $ sudo apt autoremove

                       Switch back to nouveau driver:

                           $ sudo apt install xserver-xorg-video-nouveau


                       FEDORA:
                       $ sudo dnf install ocl-icd-devel

                */
            Properties properties = new Properties();
            String osName = System.getProperty("os.name");
            if ( osName.toLowerCase().contains("linux") ) {
                String[] cmd = {"/bin/sh", "-c", "cat /etc/*-release"};
                try {
                    Process p = Runtime.getRuntime().exec(cmd);
                    BufferedReader bri = new BufferedReader(new InputStreamReader(p.getInputStream()));
                    StringBuilder text = new StringBuilder();
                    String line = "";
                    while ((line = bri.readLine()) != null) {
                        text.append(line);
                        text.append("\n");
                    }
                    properties.load(new StringReader(text.toString()));

                } catch (IOException e) {

                    e.printStackTrace();
                }
            }
            else // We just assume windows for now.
            {
                try {
                    properties.load(new StringReader("NAME=\"Windows\""));
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            //---
            String foundOS = properties.getProperty("NAME").toLowerCase().replace("\"", "");
            switch ( foundOS ) {
                case "ubuntu": return Messages.OpenCL.Tips.UBUNTU;
                case "fedora": return Messages.OpenCL.Tips.FEDORA;
                case "windows": return Messages.OpenCL.Tips.WINDOWS;
            }
            return Messages.OpenCL.Tips.UNKNOWN;
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
                    "", // Should already work
                    "Try to install the latest drivers of your GPU (Or other SIMD devices)."
            ),
            UNKNOWN(
                    "Try to install the latest OpenCL runtime for your system.",
                    "If you already have an OpenCL runtime installed consider installing the latest drivers for your GPU (Or other SIMD devices)."
            );

            public final String HOW_TO_INSTALL_OPENCL;
            private final String HOW_TO_INSTALL_OPENCL_DRIVERS;

            Tips( String howToInstallOpenCL, String howToInstallDrivers ) {
                HOW_TO_INSTALL_OPENCL = howToInstallOpenCL;
                HOW_TO_INSTALL_OPENCL_DRIVERS = howToInstallDrivers;
            }

            public String bootstrapTip() {
                return !HOW_TO_INSTALL_OPENCL.isEmpty()
                                ? (HOW_TO_INSTALL_OPENCL +""+ HOW_TO_INSTALL_OPENCL_DRIVERS)
                                : ("");
            }
        }

    }

    /**
     * {@link Operation} related messages and tips.
     */
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

    /**
     *  {@link neureka.devices.Device} implementation related messages.
     */
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

    /**
     * @param withPlaceholders The {@link String} which may or may not contain placeholder in the for of "{}".
     * @param toBePutAtPlaceholders Arbitrary {@link Object}s which will be turned into
     *                              {@link String}s instead of the placeholder brackets.
     *
     * @return A {@link String} containing the actual {@link String} representations of th {@link Object}s
     *         instead of the placeholder brackets within the first argument.
     */
    private static String _format( String withPlaceholders, Object... toBePutAtPlaceholders ) {
        return MessageFormatter.arrayFormat(withPlaceholders, toBePutAtPlaceholders).getMessage();
    }

}
