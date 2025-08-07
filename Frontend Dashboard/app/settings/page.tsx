"use client"

import { useState, useActionState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Globe, Lock, Save, User, Bell, Key, ListChecks, Plus } from 'lucide-react'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

// Placeholder for a server action to save settings
async function saveSettingsAction(prevState: any, formData: FormData) {
  const name = formData.get('name') as string;
  const email = formData.get('email') as string;
  const emailNotifications = formData.get('email-notifications') === 'on';
  const smsNotifications = formData.get('sms-notifications') === 'on';
  const alertSeverity = formData.get('alert-severity') as string;
  const defaultRegion = formData.get('default-region') as string;
  const dataRetention = formData.get('data-retention') as string;
  const mfa = formData.get('mfa') === 'on';
  const sessionTimeout = formData.get('session-timeout') as string;

  console.log("Saving settings:", {
    name, email, emailNotifications, smsNotifications, alertSeverity,
    defaultRegion, dataRetention, mfa, sessionTimeout
  });

  await new Promise(resolve => setTimeout(resolve, 1000));

  if (Math.random() > 0.1) {
    return { success: true, message: "Settings saved successfully!" };
  } else {
    return { success: false, message: "Failed to save settings. Please try again." };
  }
}

export default function SettingsPage() {
  const auditLog = [
    { id: "AL001", timestamp: "2025-08-05 11:20 AM", user: "Admin User", action: "Updated 'Riverbend' alert status to Resolved" },
    { id: "AL002", timestamp: "2025-08-05 10:05 AM", user: "System", action: "New alert 'A001' generated for Riverbend" },
    { id: "AL003", timestamp: "2025-08-04 06:00 PM", user: "Admin User", action: "Issued Evacuation Order for Hillside Village" },
    { id: "AL004", timestamp: "2025-08-03 02:15 PM", user: "John Doe", action: "Updated 'Medical Response Bravo' team status to Standby" },
  ];

  const [profileName, setProfileName] = useState("Admin User");
  const [profileEmail, setProfileEmail] = useState("admin@crisisconnect.com");
  const [emailNotifs, setEmailNotifs] = useState(true);
  const [smsNotifs, setSmsNotifs] = useState(false);
  const [minAlertSeverity, setMinAlertSeverity] = useState("medium");
  const [defaultRegion, setDefaultRegion] = useState("Global");
  const [dataRetention, setDataRetention] = useState("1-year");
  const [mfaEnabled, setMfaEnabled] = useState(true);
  const [sessionTimeout, setSessionTimeout] = useState(30);

  const [state, formAction, isPending] = useActionState(saveSettingsAction, { success: false, message: '' });

  useState(() => {
    if (state?.message) {
      alert(state.message);
    }
  });

  const handleGenerateApiKey = () => {
    alert("Generating a new API key...");
    // In a real app, this would trigger an API call and display the new key
  };

  const handleViewFullAuditLog = () => {
    alert("Navigating to full audit log details.");
    // In a real app, this would navigate to a dedicated audit log page
  };

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>User Profile</CardTitle>
          <CardDescription>Manage your personal information and account settings.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4">
          <form action={formAction}>
            <div className="grid gap-2 mb-4">
              <Label htmlFor="name">Name</Label>
              <Input id="name" name="name" value={profileName} onChange={(e) => setProfileName(e.target.value)} />
            </div>
            <div className="grid gap-2 mb-4">
              <Label htmlFor="email">Email</Label>
              <Input id="email" name="email" type="email" value={profileEmail} onChange={(e) => setProfileEmail(e.target.value)} />
            </div>
            <div className="grid gap-2 mb-4">
              <Label htmlFor="role">Role</Label>
              <Input id="role" defaultValue="Administrator" disabled />
            </div>
            <Button type="submit" className="w-full" disabled={isPending}>
              <Save className="h-4 w-4 mr-2" /> {isPending ? 'Saving...' : 'Save Profile'}
            </Button>
          </form>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Notification Preferences</CardTitle>
          <CardDescription>Configure how you receive alerts and updates.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4">
          <form action={formAction}>
            <div className="flex items-center justify-between mb-4">
              <Label htmlFor="email-notifications">Email Notifications</Label>
              <Switch id="email-notifications" name="email-notifications" checked={emailNotifs} onCheckedChange={setEmailNotifs} />
            </div>
            <div className="flex items-center justify-between mb-4">
              <Label htmlFor="sms-notifications">SMS Notifications</Label>
              <Switch id="sms-notifications" name="sms-notifications" checked={smsNotifs} onCheckedChange={setSmsNotifs} />
            </div>
            <div className="grid gap-2 mb-4">
              <Label htmlFor="alert-severity">Minimum Alert Severity</Label>
              <Select name="alert-severity" value={minAlertSeverity} onValueChange={setMinAlertSeverity}>
                <SelectTrigger id="alert-severity">
                  <SelectValue placeholder="Select severity" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="low">Low</SelectItem>
                  <SelectItem value="medium">Medium</SelectItem>
                  <SelectItem value="high">High</SelectItem>
                  <SelectItem value="critical">Critical</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <Button type="submit" className="w-full" disabled={isPending}>
              <Save className="h-4 w-4 mr-2" /> {isPending ? 'Saving...' : 'Save Preferences'}
            </Button>
          </form>
        </CardContent>
      </Card>

      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle>System Settings</CardTitle>
          <CardDescription>Global configurations for the CrisisConnect system.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-6">
          <form action={formAction}>
            <div>
              <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
                <Globe className="h-5 w-5" /> General
              </h3>
              <div className="grid gap-2 mb-4">
                <Label htmlFor="default-region">Default Operational Region</Label>
                <Input id="default-region" name="default-region" value={defaultRegion} onChange={(e) => setDefaultRegion(e.target.value)} />
              </div>
              <div className="grid gap-2 mt-4 mb-4">
                <Label htmlFor="data-retention">Data Retention Policy</Label>
                <Select name="data-retention" value={dataRetention} onValueChange={setDataRetention}>
                  <SelectTrigger id="data-retention">
                    <SelectValue placeholder="Select policy" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="6-months">6 Months</SelectItem>
                    <SelectItem value="1-year">1 Year</SelectItem>
                    <SelectItem value="3-years">3 Years</SelectItem>
                    <SelectItem value="indefinite">Indefinite</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <Separator />
            <div>
              <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
                <Lock className="h-5 w-5" /> Security
              </h3>
              <div className="flex items-center justify-between mb-4">
                <Label htmlFor="mfa">Multi-Factor Authentication (MFA)</Label>
                <Switch id="mfa" name="mfa" checked={mfaEnabled} onCheckedChange={setMfaEnabled} />
              </div>
              <div className="flex items-center justify-between mt-4 mb-4">
                <Label htmlFor="session-timeout">Session Timeout (minutes)</Label>
                <Input id="session-timeout" name="session-timeout" type="number" value={sessionTimeout} onChange={(e) => setSessionTimeout(parseInt(e.target.value))} className="w-24" />
              </div>
            </div>
            <Separator />
            <div>
              <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
                <Key className="h-5 w-5" /> API Integrations
              </h3>
              <p className="text-sm text-muted-foreground mb-2">Manage API keys for external services.</p>
              <Button variant="outline" className="w-full" type="button" onClick={handleGenerateApiKey}>
                <Plus className="h-4 w-4 mr-2" /> Generate New API Key
              </Button>
            </div>
            <Separator />
            <div>
              <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
                <ListChecks className="h-5 w-5" /> Audit Log
              </h3>
              <CardDescription className="mb-4">Review all system activities and changes.</CardDescription>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Timestamp</TableHead>
                    <TableHead>User</TableHead>
                    <TableHead>Action</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {auditLog.map((log) => (
                    <TableRow key={log.id}>
                      <TableCell className="text-xs text-muted-foreground">{log.timestamp}</TableCell>
                      <TableCell className="font-medium">{log.user}</TableCell>
                      <TableCell>{log.action}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              <Button variant="outline" className="w-full mt-4" type="button" onClick={handleViewFullAuditLog}>
                View Full Audit Log
              </Button>
            </div>
            <Button type="submit" className="w-full mt-6" disabled={isPending}>
              <Save className="h-4 w-4 mr-2" /> {isPending ? 'Saving All...' : 'Save All System Settings'}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}
