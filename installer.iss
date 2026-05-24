#define MyAppName "Seamless Texture Maker"
#define MyAppVersion "3.0.0"
#define MyAppPublisher "Shubham Panchasara"
#define MyAppExeName "SEAMS.exe"

[Setup]
AppId={{DA6FB758-C976-4700-ADC3-6B1DF8D279E4}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppMutex=SeamlessTextureMaker_Mutex_DA6FB758
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
OutputDir=dist
OutputBaseFilename=SEAMS_Setup_{#MyAppVersion}
SetupIconFile=resources\icon.ico
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible
UninstallDisplayIcon={app}\icon.ico
SetupLogging=yes
VersionInfoVersion={#MyAppVersion}
PrivilegesRequired=admin
WizardStyle=modern

[InstallDelete]
Type: files; Name: "{app}\SeamlessTextureMaker.exe"

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "resources\icon.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\SEAMS"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,SEAMS}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\SEAMS"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[Code]
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: string;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    DataDir := ExpandConstant('{appdata}\SeamlessTextureMaker');
    if DirExists(DataDir) then
    begin
      if MsgBox(
        'Do you want to delete your SEAMS user data (settings, logs, cache)?',
        mbConfirmation, MB_YESNO) = IDYES then
      begin
        DelTree(DataDir, True, True, True);
      end;
    end;
  end;
end;

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
