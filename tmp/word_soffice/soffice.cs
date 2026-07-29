using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;

public static class SofficeWordShim
{
    private static string Quote(string value)
    {
        return "\"" + (value ?? "").Replace("\"", "\\\"") + "\"";
    }

    public static int Main(string[] args)
    {
        string convertTo = "";
        string outDir = "";
        string inputPath = "";

        for (int i = 0; i < args.Length; i++)
        {
            if (String.Equals(args[i], "--convert-to", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
            {
                convertTo = args[++i];
            }
            else if (String.Equals(args[i], "--outdir", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
            {
                outDir = args[++i];
            }
            else
            {
                inputPath = args[i];
            }
        }

        string exeDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
        string script = Path.Combine(exeDir, "soffice_word.ps1");
        string psArgs = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File " + Quote(script)
            + " -ConvertTo " + Quote(convertTo)
            + " -OutDir " + Quote(outDir)
            + " -InputPath " + Quote(inputPath);

        var startInfo = new ProcessStartInfo("powershell.exe", psArgs);
        startInfo.UseShellExecute = false;
        startInfo.RedirectStandardOutput = true;
        startInfo.RedirectStandardError = true;
        startInfo.CreateNoWindow = true;
        using (var process = Process.Start(startInfo))
        {
            string stdout = process.StandardOutput.ReadToEnd();
            string stderr = process.StandardError.ReadToEnd();
            process.WaitForExit();
            if (!String.IsNullOrEmpty(stdout)) Console.Out.Write(stdout);
            if (!String.IsNullOrEmpty(stderr)) Console.Error.Write(stderr);
            return process.ExitCode;
        }
    }
}
