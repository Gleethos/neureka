package neureka.devices.opencl.utility;

import neureka.backend.api.BackendContext;
import neureka.common.utility.LogUtil;
import neureka.devices.opencl.CLContext;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.StringReader;
import java.util.Properties;

public final class Messages
{
    private Messages() {/* This is a utility class! */}

    public static String clContextCreationFailed() {
        return LogUtil.format(
                "OpenCL not available!\n" +
                        "Skipped creating and adding a new '"+ CLContext.class.getSimpleName()+"' " +
                        "to the current '"+ BackendContext.class.getSimpleName()+"'...\n" +
                        findTip().bootstrapTip()
        );
    }

    public static String clContextCouldNotFindAnyDevices() {
        return LogUtil.format(
                "OpenCL could not detect any devices in the current '{}'.\n{}",
                CLContext.class.getSimpleName(),
                findTip().HOW_TO_INSTALL_OPENCL_DRIVERS
            );
    }

    public static Tips findTip()
    {
        Properties properties = new Properties();
        String osName = System.getProperty("os.name");
        if ( osName.toLowerCase().contains("linux") ) {
            String[] cmd = {"/bin/sh", "-c", "cat /etc/*-release"};
            try {
                Process p = Runtime.getRuntime().exec(cmd);
                BufferedReader bri = new BufferedReader(new InputStreamReader(p.getInputStream()));
                StringBuilder text = new StringBuilder();
                String line = "";
                while ( (line = bri.readLine() ) != null) {
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
        String foundOS = properties.getProperty("NAME");
        if ( foundOS == null || foundOS.isEmpty() )
            return Tips.UNKNOWN;

        foundOS = foundOS.toLowerCase().replace("\"", "").trim();
        switch ( foundOS ) {
            case "ubuntu":  return Tips.UBUNTU;
            case "fedora":  return Tips.FEDORA;
            case "windows": return Tips.WINDOWS;
        }
        return Tips.UNKNOWN;
    }

    public enum Tips
    {
        UBUNTU(
            "Try executing the following command to install OpenCL: 'sudo apt install ocl-icd-opencl-dev'.\n",
            "If the OpenCL runtime cannot find your GPUs, consider executing 'sudo ubuntu-drivers autoinstall'!\n"
        ),
        FEDORA(
            "Try executing the following command to install OpenCL: 'sudo dnf install ocl-icd-devel'.\n",
            "If OpenCL runtime cannot find your GPUs, consider installing or updating your device drivers!\n"
        ),
        WINDOWS(
            "", // Should already work
            "Try to install the latest drivers of your GPU (Or other SIMD devices).\n"
        ),
        UNKNOWN(
            "Try to install the latest OpenCL runtime for your system.\n",
            "If you already have an OpenCL runtime installed consider installing the latest drivers for your GPU (Or other SIMD devices).\n"
        );

        public final String HOW_TO_INSTALL_OPENCL, HOW_TO_INSTALL_OPENCL_DRIVERS;

        Tips( String howToInstallOpenCL, String howToInstallDrivers ) {
            HOW_TO_INSTALL_OPENCL = howToInstallOpenCL;
            HOW_TO_INSTALL_OPENCL_DRIVERS = howToInstallDrivers;
        }

        public String bootstrapTip() {
            return !HOW_TO_INSTALL_OPENCL.isEmpty()
                    ? (HOW_TO_INSTALL_OPENCL +""+ HOW_TO_INSTALL_OPENCL_DRIVERS)
                    : ("Make sure you have an OpenCL runtime as well as device drivers installed.\n");
        }
    }

}
