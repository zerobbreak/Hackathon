"use client"

import { useState, useActionState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Send, MessageSquare, Mail, Phone, Globe, Clock, CheckCircle, XCircle, AlertCircle, Plus, Edit, Trash2 } from 'lucide-react'
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"

// Placeholder for a server action to send a message
async function sendMessageAction(prevState: any, formData: FormData) {
  const recipient = formData.get('recipient') as string;
  const channel = formData.get('channel') as string;
  const message = formData.get('message') as string;

  console.log("Sending message:", { recipient, channel, message });

  // Simulate API call delay
  await new Promise(resolve => setTimeout(resolve, 1000));

  if (!message || message.trim() === '') {
    return { success: false, message: "Message cannot be empty." };
  }

  // Simulate success or failure
  if (Math.random() > 0.1) { // 90% success rate
    return { success: true, message: `Message sent to ${recipient} via ${channel}.` };
  } else {
    return { success: false, message: `Failed to send message to ${recipient}. Please try again.` };
  }
}

export default function CommunicationsPage() {
  const initialCommunicationHistory = [
    {
      id: "C001",
      type: "SMS",
      recipient: "Riverbend Community",
      message: "Urgent flood warning: River levels rising rapidly. Evacuate low-lying areas.",
      status: "Sent",
      delivery: "Delivered",
      timestamp: "2025-08-05 10:05 AM",
    },
    {
      id: "C002",
      type: "Email",
      recipient: "Emergency Services",
      message: "Situation update: Teams deployed to Riverbend. Medical support on standby.",
      status: "Sent",
      delivery: "Delivered",
      timestamp: "2025-08-05 09:45 AM",
    },
    {
      id: "C003",
      type: "Public Announcement",
      recipient: "Coastal Area A",
      message: "Advisory issued: Strong currents expected near the coast. Avoid water activities.",
      status: "Sent",
      delivery: "Delivered",
      timestamp: "2025-08-05 09:35 AM",
    },
    {
      id: "C004",
      type: "SMS",
      recipient: "Hillside Village",
      message: "Evacuation order: Mandatory evacuation due to landslide risk. Proceed to shelters.",
      status: "Failed",
      delivery: "Failed (Network)",
      timestamp: "2025-08-04 06:10 PM",
    },
    {
      id: "C005",
      type: "Web Notification",
      recipient: "All Users",
      message: "Flood Watch: Heavy rainfall predicted. Monitor local conditions.",
      status: "Scheduled",
      delivery: "Pending",
      timestamp: "2025-08-06 07:00 AM",
    },
  ];

  const initialMessageTemplates = [
    { id: "T001", name: "Urgent Flood Warning", content: "Urgent flood warning: River levels rising rapidly. Evacuate low-lying areas." },
    { id: "T002", name: "Evacuation Order", content: "Mandatory evacuation due to [reason]. Proceed to designated shelters." },
    { id: "T003", name: "Situation Update", content: "Crisis update: [details]. Stay safe." },
  ];

  const [history, setHistory] = useState(initialCommunicationHistory);
  const [templates, setTemplates] = useState(initialMessageTemplates);
  const [selectedTemplate, setSelectedTemplate] = useState('');
  const [messageContent, setMessageContent] = useState('');

  const [state, formAction, isPending] = useActionState(sendMessageAction, { success: false, message: '' });

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "Sent":
        return <Badge className="bg-green-500 hover:bg-green-600 text-white">{status}</Badge>
      case "Failed":
        return <Badge variant="destructive">{status}</Badge>
      case "Scheduled":
        return <Badge variant="secondary">{status}</Badge>
      default:
        return <Badge>{status}</Badge>
    }
  }

  const getDeliveryIcon = (delivery: string) => {
    switch (delivery) {
      case "Delivered":
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case "Failed (Network)":
        return <XCircle className="h-4 w-4 text-red-500" />
      case "Pending":
        return <Clock className="h-4 w-4 text-yellow-500" />
      default:
        return <AlertCircle className="h-4 w-4 text-muted-foreground" />
    }
  }

  const handleTemplateSelect = (templateId: string) => {
    setSelectedTemplate(templateId);
    const template = templates.find(t => t.id === templateId);
    if (template) {
      setMessageContent(template.content);
    }
  };

  const handleEditTemplate = (id: string) => {
    alert(`Editing template: ${id}`);
    // In a real app, this would open a modal to edit the template
  };

  const handleDeleteTemplate = (id: string) => {
    if (confirm(`Are you sure you want to delete template ${id}?`)) {
      setTemplates(templates.filter(template => template.id !== id));
      alert(`Template ${id} deleted.`);
    }
  };

  const handleAddNewTemplate = () => {
    alert("Opening form to add a new template!");
    // In a real app, this would open a modal to add a new template
  };

  // Effect to show action state message
  useState(() => {
    if (state?.message) {
      alert(state.message);
      if (state.success) {
        // Optionally update history with the new message if successful
        // For now, just clear the form
        setMessageContent('');
        setSelectedTemplate('');
      }
    }
  });

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>Send New Communication</CardTitle>
          <CardDescription>Compose and send messages to affected communities or response teams.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4">
          <form action={formAction}>
            <div className="grid gap-2 mb-4">
              <Label htmlFor="recipient">Recipient Group</Label>
              <Select name="recipient">
                <SelectTrigger id="recipient">
                  <SelectValue placeholder="Select a group" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all-affected">All Affected Communities</SelectItem>
                  <SelectItem value="riverbend">Riverbend Community</SelectItem>
                  <SelectItem value="emergency-services">Emergency Services</SelectItem>
                  <SelectItem value="response-teams">All Response Teams</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="grid gap-2 mb-4">
              <Label htmlFor="channel">Communication Channel</Label>
              <Select name="channel">
                <SelectTrigger id="channel">
                  <SelectValue placeholder="Select a channel" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="sms">
                    SMS <MessageSquare className="inline-block ml-2 h-4 w-4" />
                  </SelectItem>
                  <SelectItem value="email">
                    Email <Mail className="inline-block ml-2 h-4 w-4" />
                  </SelectItem>
                  <SelectItem value="public-announcement">
                    Public Announcement <Phone className="inline-block ml-2 h-4 w-4" />
                  </SelectItem>
                  <SelectItem value="web-notification">
                    Web Notification <Globe className="inline-block ml-2 h-4 w-4" />
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="grid gap-2 mb-4">
              <Label htmlFor="template">Use Template</Label>
              <Select value={selectedTemplate} onValueChange={handleTemplateSelect}>
                <SelectTrigger id="template">
                  <SelectValue placeholder="Select a message template" />
                </SelectTrigger>
                <SelectContent>
                  {templates.map((template) => (
                    <SelectItem key={template.id} value={template.id}>
                      {template.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="grid gap-2 mb-4">
              <Label htmlFor="message">Message</Label>
              <Textarea
                id="message"
                name="message"
                placeholder="Type your message here..."
                rows={5}
                value={messageContent}
                onChange={(e) => setMessageContent(e.target.value)}
              />
            </div>
            <Button type="submit" className="w-full" disabled={isPending}>
              <Send className="h-4 w-4 mr-2" /> {isPending ? 'Sending...' : 'Send Message'}
            </Button>
          </form>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Communication History</CardTitle>
          <CardDescription>Review past messages sent through CrisisConnect with delivery status.</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Type</TableHead>
                <TableHead>Recipient</TableHead>
                <TableHead>Message Snippet</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Delivery</TableHead>
                <TableHead>Timestamp</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {history.map((comm) => (
                <TableRow key={comm.id}>
                  <TableCell>{comm.type}</TableCell>
                  <TableCell>{comm.recipient}</TableCell>
                  <TableCell className="text-muted-foreground">{comm.message.substring(0, 30)}...</TableCell>
                  <TableCell>{getStatusBadge(comm.status)}</TableCell>
                  <TableCell className="flex items-center gap-1">
                    {getDeliveryIcon(comm.delivery)} {comm.delivery}
                  </TableCell>
                  <TableCell>{comm.timestamp}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle>Message Templates</CardTitle>
          <CardDescription>Create and manage reusable message templates for quick communication.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4">
            {templates.map((template) => (
              <div key={template.id} className="flex items-center justify-between border rounded-md p-3">
                <div>
                  <h4 className="font-semibold">{template.name}</h4>
                  <p className="text-sm text-muted-foreground">{template.content.substring(0, 70)}...</p>
                </div>
                <div className="flex gap-2">
                  <Button variant="ghost" size="icon" onClick={() => handleEditTemplate(template.id)}>
                    <Edit className="h-4 w-4" />
                    <span className="sr-only">Edit</span>
                  </Button>
                  <Button variant="ghost" size="icon" className="text-red-600" onClick={() => handleDeleteTemplate(template.id)}>
                    <Trash2 className="h-4 w-4" />
                    <span className="sr-only">Delete</span>
                  </Button>
                </div>
              </div>
            ))}
            <Button variant="outline" className="w-full mt-2" onClick={handleAddNewTemplate}>
              <Plus className="h-4 w-4 mr-2" /> Add New Template
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
